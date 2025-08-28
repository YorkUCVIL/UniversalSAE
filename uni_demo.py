"""
This script prepares and trains a Universal Sparse Autoencoder (USAE).
It loads configuration parameters from a yaml file, prepares the dataset,
trains SAEs with cross-model prediction,
and saves results and configuration for further analysis.

Built with the Imagenet dataset in mind, but can easily be adapted.
"""

from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import shutil
import yaml

from universal_sae.data import ImageNetActivationDataset
from universal_sae.train import train_cross_prediction_saes
from overcomplete.sae import TopKSAE
from overcomplete.sae.optimizer import CosineScheduler
from overcomplete.models import DinoV2, ViT, SigLIP
from overcomplete.config.config_loader import load_model_zoo
from overcomplete.sae.losses import (
    top_k_auxiliary_loss,
    top_k_auxiliary_loss_L1,
)


def prepare_and_train_universal_sae(
    dataset,
    sae_class,
    model_zoo,
    criterion,
    sae_params,
    nb_components,
    batch_size,
    nb_epochs,
    lr=3e-3,
    weight_decay=1e-5,
    nb_epochs_warmup=1.0,
    final_lr=1e-6,
    debug=True,
    model_name="",
    divide_norm=False,
):
    if sae_class == TopKSAE:
        assert "top_k" in sae_params, "TopKSAE requires a top_k parameter"

    total_iters = int(dataset.__len__() / batch_size * nb_epochs)
    warmup_iters = int(dataset.__len__() / batch_size * nb_epochs_warmup)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8
    )

    """ Establish a separate optimiser and scheduler for each model's encoder-decoder pair """
    saes = {}
    sae_optimizers = {}
    sae_schedulers = {}
    for model in model_zoo:
        saes[model] = sae_class(
            input_shape=model_zoo[model]["input_shape"],
            n_components=nb_components,
            device="cuda",
            **sae_params,
        )
        sae_optimizers[model] = optim.AdamW(
            saes[model].parameters(), lr=lr, weight_decay=weight_decay
        )
        sae_schedulers[model] = CosineScheduler(
            sae_optimizers[model],
            lr,
            final_lr,
            total_iters=total_iters,
            warmup_iters=warmup_iters,
            start_warmup_value=1e-6,
        )

    """ Train the USAE """
    train_fn = train_cross_prediction_saes
    logs = train_fn(
        saes,
        dataloader,
        criterion,
        sae_optimizers,
        sae_schedulers,
        nb_epochs=nb_epochs,
        clip_grad=1.0,
        monitoring=debug,
        device="cuda",
        model_name=model_name,
        model_zoo=model_zoo,
        divide_norm=divide_norm,
        seeded=True,
    )

    if debug:
        avg_sparsity = np.mean(logs["z_sparsity"][-10:])
        print(f"Final sparsity: {avg_sparsity:.4f}")
        print("\n\n")

    for model in model_zoo:
        model_zoo[model]["sae"] = saes[model]

    return model_zoo


if __name__ == "__main__":
    """Models used in our paper results - see class definitions for details on activation extraction"""
    models_used = {
        "ViT": ViT,
        "DinoV2": DinoV2,
        "SigLIP": SigLIP,
    }

    """ Hyperparameters can be specified via a config yaml file """
    config_path = "config_test.yaml"
    CONFIG, CONFIG_viz, sae_params, model_zoo = load_model_zoo(config_path, models_used)

    """
    Our data pipeline:
    - During training, each batch is expected to be in the form:
        Model Activations X and labels Y in tuple: ( X{Dict: (models s_1...s_k)}, Y)
        where each value from model key s_i is in form: N(batch size) x d_i(activation dimension)
    - Activation Dataset/Loader is customized for training on Imagenet:
        - requires model activations to be cached
        - assumes activation directory follows format of Imagenet directory
        - recommended to store all models' activations for each image in
        one npz file (hence 'combined_npz' param)
        - see models.py for model activation caching details
    """

    dataset = ImageNetActivationDataset(
        root=CONFIG["imagenet_root"],
        activation_root=CONFIG["path_to_cache"],
        combined_npz=CONFIG["combined_npz"],
        target_class=CONFIG["target_class"],
        split="train",
        standardize=CONFIG["standardize"],
        divide_norm=CONFIG["divide_norm"],
        use_class_tokens=CONFIG["use_class_tokens"],
        sources=list(model_zoo.keys()),
    )

    """ We experimented comparing L1 vs L2 loss for dict alignment, and found L1 loss to produce more distinctive features """
    if CONFIG["loss_criterion"] == "L2":
        criterion = top_k_auxiliary_loss
    else:
        criterion = top_k_auxiliary_loss_L1

    if CONFIG["standardize"]:
        for model in model_zoo:
            model_zoo[model]["model_mean"] = dataset.standardization_stats[model][
                "mean"
            ]
            model_zoo[model]["model_std"] = dataset.standardization_stats[model]["std"]

    model_zoo = prepare_and_train_universal_sae(
        dataset=dataset,
        sae_class=TopKSAE,
        criterion=criterion,
        model_zoo=model_zoo,
        sae_params=sae_params,
        lr=CONFIG["lr"],
        final_lr=CONFIG["final_lr"],
        nb_components=CONFIG["nb_components"],
        batch_size=CONFIG["batch_size"],
        nb_epochs=CONFIG["nb_epochs"],
        debug=CONFIG["debug"],
        model_name=CONFIG["run_name"],
        divide_norm=CONFIG["divide_norm"],
    )

    """
    After training save config file to results directory
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    yaml_name = os.path.basename(config_path)
    results_yaml = os.path.join(results_dir, yaml_name)
    shutil.copy(config_path, results_yaml)

    # edit the std and mean values for each model to config file
    with open(results_yaml) as f:
        yaml_content = yaml.safe_load(f)
    if CONFIG["standardize"]:
        for model, params in model_zoo.items():
            yaml_content["model_zoo"][model]["model_mean"] = float(params["model_mean"])
            yaml_content["model_zoo"][model]["model_std"] = float(params["model_std"])
            yaml_content["model_zoo"][model]["checkpoint_path"] = params[
                "checkpoint_path"
            ]

    yaml_content["global"] = CONFIG
    yaml_content["viz"] = CONFIG_viz
    yaml_content["sae_params"] = sae_params

    with open(results_yaml, "w") as file:
        yaml.dump(yaml_content, file, default_flow_style=False, sort_keys=False)

    print(f"Results stored in: {results_dir}")
