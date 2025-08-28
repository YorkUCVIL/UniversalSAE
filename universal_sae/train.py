"""
Module for training Universal Sparse Autoencoder (USAE) models.
"""

import time
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from einops import rearrange
import wandb
import numpy as np
import random

from universal_sae.uni_visuals import (
    plot_reconstruction_matrix,
    interpolate_patch_tokens,
)
from overcomplete.metrics import l2, sparsity, r2_score


def save_checkpoint(model, optimizer, epoch, lr, path="checkpoint.pth"):
    # Create the checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate": lr,
    }

    # Extract the directory from the checkpoint path
    dir_path = os.path.dirname(path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created at: {dir_path}")

    # Save the checkpoint
    torch.save(checkpoint, path)


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.

    If the input (x) shape is 4D assume it is (n, c, w, h), if it is 3D assume
    it is (n, t, c). Else, assume it is already flattened.

    Concerning the reconstruction error, we want a measure that could be compared
    across different input shapes, so we use the R2 score: 1 means perfect
    reconstruction, 0 means no reconstruction at all.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.

    Returns
    -------
    float
        Reconstruction error.
    """
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)

    return r2.item()


def _log_metrics(monitoring, logs, model, z, loss, optimizer):
    """
    Log training metrics for the current training step.

    Parameters
    ----------
    monitoring : int
        Monitoring level, for 1 store only basic statistics, for 2 store more detailed statistics.
    logs : defaultdict
        Logs of training statistics.
    model : nn.Module
        The SAE model.
    z : torch.Tensor
        Encoded tensor.
    loss : torch.Tensor
        Loss value.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    """
    if monitoring == 0:
        return

    if monitoring > 0:
        if isinstance(optimizer, dict):
            for model in optimizer.keys():
                logs[f"{model}_lr"].append(optimizer[model].param_groups[0]["lr"])
        else:
            logs["lr"].append(optimizer.param_groups[0]["lr"])

        logs["step_loss"].append(loss.item())

    if monitoring > 1:  # appending tensor for monitoring>1 grows ram quickly
        # store directly some z values
        # and the params / gradients norms
        logs["z"] = z.detach().cpu()[::10]
        logs["z_l2"].append(l2(z).item())

        logs["dictionary_sparsity"].append(
            sparsity(model.get_dictionary()).mean().item()
        )
        logs["dictionary_norms"].append(l2(model.get_dictionary(), -1).mean().item())

        for name, param in model.named_parameters():
            if param.grad is not None:
                logs[f"params_norm_{name}"].append(l2(param).item())
                logs[f"params_grad_norm_{name}"].append(l2(param.grad).item())


def _avg_non_diagonal(arr):  # used for HP-Sweep to max R2 cross entries
    """
    Calculate mean of non-diagonal elements in R2 matrix

    Parameters
    ----------
    arr (numpy.ndarray): 2D input array

    Returns
    -------
    float: mean of non-diagonal elements
    """
    mask = ~np.eye(arr.shape[0], arr.shape[1], dtype=bool)
    return np.mean(arr[mask])


def train_cross_prediction_saes(
    saes,
    dataloader,
    criterion,
    sae_optimizers,
    sae_schedulers,
    model_zoo,
    nb_epochs=20,
    clip_grad=1.0,
    monitoring=1,
    device="cpu",
    seeded=True,
    model_name="",
    divide_norm=False,
):
    """
    Parameters
    ----------
    *saes : dict: SAE(nn.Module)
        dict of sae modules which have been pre-initialized according to each model.
    *dataloader : DataLoader
        DataLoader providing the training data, assumed to be activations (& labels) derived from K models.
        Each sample's activation data comprises activations from all models (Sample j -> Model_1(j), Model_2(j)... )
    criterion : callable
        Loss function.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
         (0) silent.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cpu'.
    seeded : whether or not a random seed is set, ensuring reproducibility

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    logs = defaultdict(list)
    wandb.init(
        name=f"cross_prediction_usae_{model_name}",
        project="universal-sae",
    )

    if seeded:  # for replicability
        random.seed(10)

    # multiply by 100 to adjust loss if L1 with Norm 1
    norm_correct = 1
    if divide_norm and criterion.__name__ == "top_k_auxiliary_loss_L1":
        norm_correct = 100

    for epoch in range(nb_epochs):
        # prep modules for training
        for sae in saes.values():
            sae.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0

        # reconstructions table: logging reconstrution accuracy
        # rows: Encoded activations z_i from model i. cols: model activation and decoder j used for reconstruction
        reconstruction_table = np.zeros((len(saes), len(saes)), dtype=float)
        update_count = np.zeros((len(saes), len(saes)), dtype=int)

        # Model Activations X and labels Y are in tuple: ( X{Dict: (models s_1...s_k)}, Y)
        # where each value from model key s_i is in form: N(batch size) x d_i(activation dimension)
        for i, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="epoch progress",
            dynamic_ncols=True,
        ):
            batch_set, _ = batch

            # select i, model index for encoding activations A_i to z with Encoder i
            i = random.randint(0, len(batch_set) - 1)
            model_i = list(batch_set.keys())[i]

            # important to zero this optimizer's gradients, as previous forward passes will have changed them
            sae_optimizers[model_i].zero_grad()

            # Encoding #
            A_i = batch_set[model_i].to(device, non_blocking=True)

            # This is somewhat hardcoded to account for DinoV2/CLIP activations, which use 196 tokens instead of 256
            if model_i in {
                "DinoV2",
                "CLIP",
            }:
                A_i = interpolate_patch_tokens(
                    A_i, num_patches_in=16, num_patches_out=14
                )

            # Rearrange to 2D tensor
            z_pre, z_i = saes[model_i].encode(rearrange(A_i, "b n c -> (b n) c"))

            # Decoding z to all decoders #
            loss = 0.0
            for j, model in enumerate(saes.keys()):
                if model == model_i:  # don't reload (and interpolate) same batch
                    A = A_i
                else:
                    A = batch_set[model].to(device, non_blocking=True)
                    if model in {"DinoV2", "CLIP"}:
                        A = interpolate_patch_tokens(
                            A, num_patches_in=16, num_patches_out=14
                        )

                A_hat = saes[model].decode(z_i)

                loss += (
                    criterion(
                        rearrange(A, "b n c -> (b n) c"),
                        A_hat,
                        z_pre,
                        z_i,
                        saes[model].get_dictionary(),
                    )
                    * norm_correct
                )

                if monitoring:
                    r2_error = _compute_reconstruction_error(A, A_hat)
                    update_count[i, j] += 1
                    reconstruction_table[i, j] += r2_error

            # compute backwards loss after all losses summed
            loss.backward()

            if clip_grad:
                for sae in saes.values():
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)

            # note only model i encoder/decoder weights are changed
            sae_optimizers[model_i].step()

            if sae_schedulers[model_i] is not None:
                sae_schedulers[model_i].step()

            if monitoring:
                epoch_loss += loss.item()
                epoch_error += reconstruction_table.sum()
                epoch_sparsity += sparsity(z_i).mean().item()
                _log_metrics(monitoring, logs, saes, z_i, loss, sae_optimizers)
                log_data = {key: value[-1] for key, value in logs.items()}
                wandb.log(log_data)

        if monitoring:
            avg_loss = epoch_loss / len(dataloader)
            avg_error = epoch_error / (len(dataloader) * len(list(model_zoo.keys())))
            avg_sparsity = epoch_sparsity / len(dataloader)
            epoch_duration = time.time() - start_time

            logs["avg_loss"].append(avg_loss)
            logs["time_epoch"].append(epoch_duration)
            logs["z_sparsity"].append(avg_sparsity)
            log_data = {key: value[-1] for key, value in logs.items()}
            wandb.log(log_data)

            # log reconstruction matrix
            avg_table = np.asarray(reconstruction_table / (update_count + 1e-10))
            fig = plot_reconstruction_matrix(avg_table, list(saes.keys()))
            wandb.log({"reconstruction_matrix_plot": fig})

            cross_R2 = _avg_non_diagonal(avg_table)
            wandb.log({"cross_R2": cross_R2})

            print(
                f"Epoch[{epoch+1}/{nb_epochs}], Loss: {avg_loss:.4f}, "
                f"R2: {avg_error:.4f}, Sparsity: {avg_sparsity:.4f}, "
                f"Time: {epoch_duration:.4f} seconds"
            )

        # Save checkpoints
        if epoch % 10 == 0 or epoch + 1 == nb_epochs:
            for model in saes:
                model_n = model.split(".")[0]
                checkpoint_path = (
                    os.getcwd()
                    + f"/weights/{model_name}/{model_n}/uni_{model_n}_checkpoint_{epoch}.pth"
                )
                model_zoo[model]["checkpoint_path"] = checkpoint_path

                save_checkpoint(
                    model=saes[model],
                    optimizer=sae_optimizers[model],
                    epoch=epoch,
                    lr=sae_optimizers[model].param_groups[0]["lr"],
                    path=checkpoint_path,
                )

    wandb.finish()
    return logs
