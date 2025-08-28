import matplotlib.pyplot as plt
import torch
from einops import rearrange


def plot_reconstruction_matrix(values, model_names, min=0, max=1):
    """
    Creates a matplotlib heatmap, assuming r2 reconstruction matrix values between models

    Parameters
    ----------
    values : torch.Tensor or ndarray
            Input of shape (len(model_names), len(model_names)).
    model_names : list of strings
            A list of model names
    min : float or int, optional
            Min value of color mapping, default = 0
    max : float or int, optional
            Max value of color mapping, default = 1

    Returns
    -------
    fig : pyplot fig
            Returns figure

    """
    assert len(values) == len(model_names)
    NUM_MODELS = len(values)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(
        values, cmap="viridis", interpolation="nearest", vmin=min, vmax=max
    )
    ax.set_title(
        "Reconstruction Matrix ($R^2$-score)", fontsize=16, fontweight="medium", y=-0.18
    )
    ax.set_xlabel(
        "Decoder and Activations $j$ used for reconstruction",
        fontweight="medium",
        fontsize=14,
        labelpad=15,
    )
    ax.set_ylabel(
        "Model Activations $i$ Encoded to $Z$", fontweight="medium", fontsize=14
    )
    plt.xticks(
        ticks=range(0, NUM_MODELS), labels=model_names, fontweight="medium", fontsize=14
    )
    plt.yticks(
        ticks=range(0, NUM_MODELS), labels=model_names, fontweight="medium", fontsize=14
    )
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    for i in range(NUM_MODELS):
        for j in range(NUM_MODELS):
            plt.text(
                j,
                i,
                f"{values[i][j]:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize="large",
                color="white",
            )
    plt.colorbar(heatmap)
    plt.tight_layout()
    return fig


def interpolate_patch_tokens(A, num_patches_in, num_patches_out):
    """
    Interpolates activations to a new patch token resolution.
    ex. to interpolate 256 patch tokens to 196 patch tokens, then num_patches_in = 16, num_patches_out=14

    Parameters
    ----------
    A : torch.Tensor
            Input tensor of shape (BATCH, NUMTOKENS, CHANNEL) containing class and patch tokens.
    num_class_tokens : int   *** removed: INFERRED by A.shape ***
            Number of class tokens in the tensor (kept unchanged).
    num_patches_in : int
            Current number of patch tokens (height or width of patch grid).
    num_patches_out : int
            Target number of patch tokens (height or width of new patch grid).

    Returns
    -------
    torch.tensor
            interpolated activations, back into form BATCH x NUMTOKENS x CHANNEL
    """
    num_class_tokens = A.shape[1] - num_patches_in**2
    patches = A[:, num_class_tokens:, :]  # keep the patch tokens
    patches = rearrange(
        patches, "n (h w) c -> n c h w", h=num_patches_in, w=num_patches_in
    )
    patches_interp = torch.nn.functional.interpolate(
        patches,
        size=(num_patches_out, num_patches_out),
        mode="bilinear",
        antialias=True,
    )
    interp = rearrange(patches_interp, "n c h w -> n (h w) c ")

    if num_class_tokens > 0:
        cls = A[:, num_class_tokens - 1, :].unsqueeze(dim=1)  # grab the cls token
        interp = torch.cat((cls, interp), dim=1)

    return interp
