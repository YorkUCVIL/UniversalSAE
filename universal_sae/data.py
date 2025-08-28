"""
Module for loading Imagenet Activations Dataset
"""

import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from torchvision.datasets import ImageNet
from typing import Tuple, Dict, Union


class ImageNetActivationDataset(ImageNet):
    def __init__(
        self,
        root: str,
        activation_root: str,
        sources: list,
        combined_npz: bool = False,
        split: str = "train",
        target_class: Union[str, int, None] = None,
        standardize: bool = False,
        divide_norm: bool = False,
        use_class_tokens: bool = True,
        **kwargs,
    ):
        """
        Dataset for loading ImageNet activations with optional class filtering.

        Args:
                root: Root directory of the ImageNet dataset
                activation_root: Root directory containing activation files
                sources: List of models used (to retrieve activation files)
                combined_npz:  whether the derived model activations from each
                        image are stored all together in the same npz file (for quicker loading)
                split: 'train' or 'val'
                target_class: Can be either:
                        - ImageNet class index (int, 0-999)
                        - ImageNet class name (str: ex. "tabby cat")
                        - None to load all classes
                standardize: Whether to standardize activations using mean/std
                divide_norm: Whether to normalize activations by their L2 norm
                use_class_tokens: Whether class tokens will be included in training
        """
        # Initialize parent ImageNet dataset
        super().__init__(root=root, split=split, **kwargs)

        # Filter to target class if specified
        if target_class is not None and target_class != "ALL":
            if isinstance(target_class, str):
                if target_class not in self.class_to_idx:
                    raise ValueError(f"Invalid WordNet ID: {target_class}")
                target_class = self.class_to_idx[target_class]
            # Filter ImageNet samples to only target class
            self.samples = [
                (path, idx) for path, idx in self.samples if idx == target_class
            ]
            print(f"Filtered to {len(self.samples)} samples for class {target_class}")
        else:
            print("Training on all classes")

        self.activation_root = activation_root
        self.sources = sources
        self.combined_npz = combined_npz
        self.standardize = standardize
        self.divide_norm = divide_norm
        self.split = split
        self.use_class_tokens = use_class_tokens

        # Get act_paths from self.samples outside of getitem
        if self.combined_npz:
            self.samples_used = []
            for img_path, target in tqdm(self.samples, total=len(self.samples)):
                rel_path = os.path.relpath(
                    img_path, os.path.join(self.root, self.split)
                )
                class_dir = os.path.dirname(rel_path)
                if self.combined_npz is not None:
                    act_path = os.path.join(
                        self.activation_root,
                        self.split,
                        class_dir,
                        os.path.basename(img_path).replace(".JPEG", "_combined.npz"),
                    )
                    self.samples_used.append((act_path, target))
        else:
            self.samples_used = {}
            for source in self.sources:
                print("collecting act_paths")
                self.samples_used[source] = []
                for img_path, target in tqdm(self.samples, total=len(self.samples)):
                    rel_path = os.path.relpath(
                        img_path, os.path.join(self.root, self.split)
                    )
                    class_dir = os.path.dirname(rel_path)
                    act_path = os.path.join(
                        self.activation_root,
                        self.split,
                        class_dir,
                        os.path.basename(img_path).replace(
                            ".JPEG", "_" + source + ".npz"
                        ),
                    )
                    self.samples_used[source].append((act_path, target))

        # Calculate standardization stats if needed
        if self.standardize:
            self._compute_standardization_stats()

    def _compute_standardization_stats(self, sample_size: int = 1000):
        """Compute sample mean and std of activations for standardization"""
        self.standardization_stats = {}

        sample_size = min(sample_size, len(self.samples))
        sample_indices = np.random.choice(len(self.samples), sample_size, replace=False)

        for source in self.sources:
            print(f"Computing standardization stats for {source}")

            activations = []
            for idx in tqdm(sample_indices):
                if self.combined_npz:
                    act_path, target = self.samples_used[idx]
                    act = torch.from_numpy(np.load(act_path)[source])
                else:
                    act_path, target = self.samples_used[source][idx]
                    act = torch.tensor(np.load(act_path)["activation"])

                if source in {"DinoV2", "ViT", "CLIP"}:  # (models that use class token)
                    if not self.use_class_tokens:
                        act = act[1:, :]
                activations.append(act)

            activations = torch.cat(activations, dim=0)
            mean = activations.mean(dim=(0, 1))
            std = activations.std(dim=(0, 1))
            print("MEAN: ", mean)
            print("STD: ", std)
            self.standardization_stats[source] = {"mean": mean, "std": std}

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Returns
        -------
                tuple: (activations, target) where activations is a
                dictionary of activations for each source, and target is the
                class label index.
        """
        if self.combined_npz:
            act_path, target = self.samples_used[index]
            npz_file = np.load(act_path, mmap_mode="r")
            act_dict = {key: torch.from_numpy(npz_file[key]) for key in npz_file.files}

        activations = {}
        for source in self.sources:
            if self.combined_npz:
                act = act_dict[source]
            else:
                act_path, target = self.samples_used[source][index]
                act = torch.from_numpy(np.load(act_path, mmap_mode="r")["activation"])

            if source in {"DinoV2", "ViT", "CLIP"}:  # (models that use class token)
                if not self.use_class_tokens:
                    act = act[1:, :]

            # standardize/normalize activations
            if self.standardize:
                mean = self.standardization_stats[source]["mean"]
                std = self.standardization_stats[source]["std"]
                act = (act - mean) / (std + 1e-5)
            elif self.divide_norm:
                act = act / act.norm(dim=-1, keepdim=True)

            activations[source] = act
        return activations, target

    def __len__(self) -> int:
        return len(self.samples)


def load_directory(directory):
    """
    Load all images from a directory.

    Parameters
    ----------
    dir : str
            Directory path.

    Returns
    -------
    list
            List of PIL images.
    """
    paths = os.listdir(directory)
    paths = [path for path in paths if path.endswith((".jpg", ".jpeg", ".png"))]
    paths = sorted(paths)

    images = []
    for path in paths:
        try:
            img = Image.open(os.path.join(directory, path)).convert("RGB")
            images.append(img)
        except OSError:
            # skip files that are not images
            continue
    return images
