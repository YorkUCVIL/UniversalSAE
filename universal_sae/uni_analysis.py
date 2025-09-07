from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import os
from overcomplete.metrics import (
	dead_codes,
	energy_of_codes,
	r2_score,
)
from overcomplete.visualization.cmaps import VIRIDIS_ALPHA, TAB10_ALPHA
from overcomplete.visualization.top_concepts import _get_representative_ids
from overcomplete.visualization.plot_utils import (
	show,
	interpolate_cv2,
	get_image_dimensions,
	np_channel_last,
	normalize,
)

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


def unwrap_dataloader(dataloader):
	"""
	Unwrap a DataLoader into a single tensor.

	Parameters
	----------
	dataloader : DataLoader
		DataLoader object.
	"""
	return torch.cat([batch[0] if isinstance(batch, (tuple, list))
					  else batch for batch in dataloader], dim=0)


@dataclass
class ActivationStats:
    """Container for activation statistics."""
    sums: torch.Tensor
    sums_thresh: torch.Tensor
    total_fires: int
    fires_above_threshold: int
    dead_concepts_pct: float
    dead_concepts_thresh_pct: float


@dataclass
class CofiringStats:
    """Container for cofiring statistics."""
    matrix: torch.Tensor
    total: float
    max_value: float
    mean: float


class ModelActivationExtractor:
    """Extracts activations from different vision model architectures."""

    def __init__(self, config: Dict, config_viz: Dict, model_zoo: Dict):
        """
        Initialize the activation extractor.

        Args:
            config: General configuration dictionary
            config_viz: Visualization configuration dictionary
            model_zoo: Dictionary mapping model names to model configurations
        """
        self.config = config
        self.config_viz = config_viz
        self.model_zoo = model_zoo
        self.model_names = list(model_zoo.keys())

    def create_class_dataloader(self, transform):
        """Create dataloader for a single ImageNet class"""

        dataset = datasets.ImageNet(root='/local/ssd/harryt/datasets/imagenet', split='val', transform=transform)

        if self.config_viz['class_name'] != 'ALL':
            # Filter for specific class
            print(f"Loading images for class: {self.config_viz['class_name']}")
            class_idx = dataset.class_to_idx.get(self.config_viz['class_name'])
            class_indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
            subset_indices = class_indices[:self.config_viz['sample_size']]

        else:
            # Random subset from all classes
            #subset_indices = torch.randperm(len(dataset))[:self.config_viz['sample_size']].tolist()
            subset_indices = torch.arange(len(dataset))[:self.config_viz['sample_size']].tolist()

        subset = Subset(dataset, subset_indices)
        return DataLoader(subset, batch_size=self.config_viz['batch_size'], shuffle=False)

    def _reshape_activations(self, activations: torch.Tensor, model_class: str) -> torch.Tensor:
        """
        Reshape activations based on model architecture.

        Args:
            activations: Raw model activations
            model_class: Name of the model class

        Returns:
            Reshaped activations tensor
        """
        if model_class == "ViT":
            if not self.config['use_class_tokens']:
                activations = activations[:, 1:, :]  # Remove class token
                assert activations.shape[1] == 196, f"Expected 196 patches, got {activations.shape[1]}"
            return rearrange(activations, "n t d -> (n t) d")

        elif model_class in {"DinoV2"}:
            if not self.config['use_class_tokens']:
                activations = activations[:, 1:, :]  # Remove class token
                assert activations.shape[1] == 256, f"Expected 256 patches, got {activations.shape[1]}"

            # Interpolate patch tokens for different resolutions
            activations = interpolate_patch_tokens(
                activations, num_patches_in=16, num_patches_out=14
            )
            return rearrange(activations, "n t d -> (n t) d")

        elif model_class == 'SigLIP':
            return rearrange(activations, "n t d -> (n t) d")

        else:
            raise ValueError(f"Unsupported model class: {model_class}")


    def _normalize_activations(self, activations: torch.Tensor, model_config: Dict) -> torch.Tensor:
        """
        Apply normalization to activations based on configuration.

        Args:
            activations: Input activations
            model_config: Model-specific configuration

        Returns:
            Normalized activations
        """
        if self.config['standardize']:
            mean = model_config['model_mean']
            std = model_config['model_std']
            return (activations - mean) / (std + 1e-5)

        elif self.config['divide_norm']:
            return activations / activations.norm(dim=-1, keepdim=True)

        return activations



    def extract_all_activations(self) -> Dict[str, torch.Tensor]:
        """
        Extract and process activations from all models in the model zoo.

        Returns:
            Dictionary mapping model names to their activations
        """
        model_activations = {}

        for model_name in self.model_zoo.keys():
            print(f"Computing activations for {model_name}")

            model_config = self.model_zoo[model_name]
            model = model_config['og_model'](device='cuda').float().eval()

            # Create dataloader
            data_loader = self.create_class_dataloader(transform=model.preprocess)

            # Extract activations
            all_activations = []
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f"Processing {model_name}"):
                    activations = model.forward_features(batch[0].cuda())

                    activations = self._reshape_activations(
                        activations, model.__class__.__name__
                    )

                    activations = self._normalize_activations(activations, model_config)
                    all_activations.append(activations.detach().cpu())

            model_activations[model_name] = torch.cat(all_activations, dim=0)

        return model_activations, data_loader


class SAEAnalyzer:
    """Analyzes Sparse Autoencoder outputs and computes reconstruction metrics."""

    def __init__(self, model_zoo: Dict, config: Dict, config_viz: Dict,
                 process_heatmaps: bool = True):
        """
        Initialize the SAE analyzer.

        Args:
            model_zoo: Dictionary of model configurations (each with SAE pair)
            config: General configuration
            config_viz: Visualization configuration
            process_heatmaps: Whether to process spatial heatmaps
        """
        self.model_zoo = model_zoo
        self.config = config
        self.config_viz = config_viz
        self.process_heatmaps = process_heatmaps

    def compute_sae_statistics(self, model_activations: Dict[str, torch.Tensor]) -> Dict:
        """
        Compute SAE statistics for all models.

        Args:
            model_activations: Dictionary mapping model names to their activations

        Returns:
            Dictionary of statistics for each model
        """
        results = {}
        for model_name in self.model_zoo.keys():
            print(f"Computing SAE stats for {model_name}")
            results[model_name] = self._compute_model_stats(
                model_name, model_activations[model_name]
            )
        return results

    def _compute_model_stats(self, model_name: str, activations: torch.Tensor) -> Dict:
        """Compute statistics for a specific model."""
        model_config = self.model_zoo[model_name]
        num_tokens = model_config['num_tokens']

        # Initialize storage for latent activations
        total_samples = self.config_viz['sample_size'] * num_tokens
        z_activations = torch.zeros((total_samples, self.config['nb_components'])).cpu()

        # Process in batches
        stats = self._process_activation_batches(
            activations, model_config, num_tokens, z_activations
        )

        # Add configuration info
        stats['config'] = {
            'model_name': model_name,
            'input_shape': model_config['input_shape'],
            'num_tokens': num_tokens
        }

        return stats

    def _process_activation_batches(self, activations: torch.Tensor, model_config: Dict,
                                  num_tokens: int, z_activations: torch.Tensor) -> Dict:
        """Process activations in batches and compute reconstruction metrics."""
        batch_size = self.config_viz['batch_size']
        batch_datapoints = num_tokens * batch_size
        num_batches = len(activations) // batch_datapoints

        metrics = {
            'heatmaps': [],
            'dead_codes': [],
            'mse_losses': [],
            'r2_scores': [],
            'energies': []
        }

        sae = model_config['sae']

        for i in tqdm(range(num_batches), desc="Processing batches", leave=False):
            start_idx = i * batch_datapoints
            end_idx = (i + 1) * batch_datapoints

            batch_activations = activations[start_idx:end_idx].cuda()

            # Encode to sparse representation
            z_pre_act, z_activations_batch = sae.encode(batch_activations)
            z_activations[start_idx:end_idx, :] = z_activations_batch.detach().cpu()

            # Create spatial heatmaps
            if self.process_heatmaps:
                spatial_dim = int(np.sqrt(num_tokens))
                heatmap = z_activations_batch.reshape(
                    (batch_size, spatial_dim, spatial_dim, -1)
                ).detach().cpu()
                metrics['heatmaps'].append(heatmap)

            # Decode back to original space
            reconstructions = sae.decode(z_activations_batch).detach().cpu()
            targets = activations[start_idx:end_idx]

            # Compute metrics
            metrics['dead_codes'].append(dead_codes(z_activations_batch).detach().cpu())
            metrics['mse_losses'].append(
                (targets - reconstructions).square().mean().detach().cpu()
            )
            metrics['r2_scores'].append(r2_score(targets, reconstructions).detach().cpu())
            metrics['energies'].append(
                energy_of_codes(z_activations_batch, sae.get_dictionary()).detach().cpu()
            )

            del z_pre_act, z_activations_batch, batch_activations, reconstructions

        # Aggregate results
        results = {
            'z_activations': z_activations,
            'dead': torch.stack(metrics['dead_codes']),
            'mse': torch.stack(metrics['mse_losses']),
            'r2': torch.stack(metrics['r2_scores']),
            'energies': torch.stack(metrics['energies']).mean(0)
        }

        if self.process_heatmaps:
            results['heatmaps'] = rearrange(
                torch.stack(metrics['heatmaps']), "b n w h c -> (b n) w h c"
            )

        return results


class ActivationAnalyzer:
    """Analyzes activation patterns and computes alignment metrics between models."""

    def __init__(self, results: Dict, config: Dict, config_viz: Dict, model_zoo: Dict):
        """
        Initialize the activation analyzer.

        Args:
            results: Dictionary of SAE analysis results
            config: General configuration
            config_viz: Visualization configuration
            model_zoo: Dictionary of model configurations
        """
        self.results = results
        self.config = config
        self.config_viz = config_viz
        self.model_zoo = model_zoo
        self.model_names = list(model_zoo.keys())

        # Extract z_activations from results
        self.z_activations = {
            name: results[name]['z_activations'] for name in self.model_names
        }
        self.alignment_metrics = {}

    def _apply_activation_threshold(self, threshold: float) -> Dict[str, torch.Tensor]:
        """Apply threshold to activations, setting values below threshold to zero."""
        return {
            name: torch.where(z > threshold, z, 0)
            for name, z in self.z_activations.items()
        }

    def _compute_activation_sums(self, z_activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute sum of non-zero activations for each concept."""
        return {name: z.bool().sum(dim=0) for name, z in z_activations.items()}

    def _compute_activation_statistics(self, sums: Dict[str, torch.Tensor],
                                     sums_thresh: Dict[str, torch.Tensor]) -> Dict[str, ActivationStats]:
        """Compute basic activation statistics for each model."""
        stats = {}
        num_concepts = self.config['nb_components']

        for name in self.model_names:
            dead_count = torch.count_nonzero(sums[name] == 0)
            dead_thresh_count = torch.count_nonzero(sums_thresh[name] == 0)

            stats[name] = ActivationStats(
                sums=sums[name],
                sums_thresh=sums_thresh[name],
                total_fires=sums[name].sum().item(),
                fires_above_threshold=sums_thresh[name].sum().item(),
                dead_concepts_pct=100.0 * dead_count / num_concepts,
                dead_concepts_thresh_pct=100.0 * dead_thresh_count / num_concepts
            )
        return stats

    def _compute_cofiring_matrix(self, z_activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cofiring matrix across all models - simplified for small datasets."""
        z_boolean = {name: z.bool() for name, z in z_activations.items()}

        # compute intersection (AND operation) across all models
        # Stack all model activations and compute element-wise AND
        all_activations = torch.stack([z for z in z_boolean.values()])  # [num_models, samples, concepts]
        cofiring = all_activations.all(dim=0).float()  # [samples, concepts]

        return cofiring

    def compute_cofiring_statistics(self, z_activations: Dict[str, torch.Tensor]) -> CofiringStats:
        """Compute cofiring statistics across all models."""
        matrix = self._compute_cofiring_matrix(z_activations)

        return CofiringStats(
            matrix=matrix,
            total=matrix.sum().item(),
            max_value=matrix.max().item(),
            mean=matrix.mean().item()
        )

    def compute_cofiring_statistics(self, z_activations: Dict[str, torch.Tensor],
                        batch_div: int = 10) -> Dict[str, CofiringStats]:
        matrix = self._compute_cofiring_matrix(z_activations)

        cofiring_stats = CofiringStats(
            matrix=matrix,
            total=matrix.sum().item(),
            max_value=matrix.max().item(),
            mean=matrix.mean().item()
        )

        return cofiring_stats

    #scaling to larger val set, comute cofiring in batches
    #def _compute_cofiring_matrix(self, z_activations: Dict[str, torch.Tensor],
    #                           batch_div: int) -> torch.Tensor:
    #    """Compute cofiring matrix for a pair of activations"""

    #    z_bools = {name: z.bool() for name, z in z_activations.items()}

    #    total_samples = z_bools[list(z_bools.keys())[0]].shape[0] #num samples for model 1
    #    assert total_samples % batch_div == 0
    #    batch_size = total_samples // batch_div

    #    cofiring = []
    #    for batch_idx in range(batch_div):
    #        start_idx = batch_idx * batch_size
    #        end_idx = (batch_idx + 1) * batch_size

    #        batch = self._compute_batch_cofiring(
    #            start_idx,
    #            end_idx,
    #            z_bools,
    #        )
    #        cofiring.append(batch)

    #    cofiring = torch.stack(cofiring)#.sum(dim=0) #[batch_div, batch_size, num_concepts]
    #    cofiring = cofiring.view(total_samples, -1)#.sum(dim=0)

    #    #print(cofiring.shape)
    #    return cofiring

    #@staticmethod
    #def _compute_batch_cofiring(start_idx: int, end_idx: int, z_bools: Dict[str, torch.Tensor]) -> torch.Tensor:
    #    """Compute cofiring for a single batch"""
    #    #print(z1.shape,z2.shape, (z1&z2).shape)
    #    #below if equivalent to computing an intersection
    #    #return (z1.cuda() @ z2.cuda()).cpu().float() #TODO update for outerproduct if needed
    #    return (torch.stack([z[start_idx:end_idx] for z in z_bools.values()]).all(dim=0)).cpu().float()

    def compute_cofire_iou_metric(self, cofiring_matrix: torch.Tensor,
                                union_fires: torch.Tensor, energies: torch.Tensor,
                                sort_by_energy: bool = True, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU) metric for cofiring.

        Args:
            cofiring_matrix: Matrix of cofiring events [samples, concepts]
            union_fires: Total fires per concept across all models
            energies: Energy values for sorting concepts
            sort_by_energy: Whether to sort concepts by energy
            eps: Small epsilon for numerical stability

        Returns:
            IoU values per concept
        """
        if sort_by_energy:
            concept_order = energies.argsort(descending=True)
        else:
            concept_order = torch.arange(energies.shape[0])

        sorted_cofire_sums = cofiring_matrix[:, concept_order].sum(dim=0)
        iou = sorted_cofire_sums / (union_fires[concept_order] + eps)

        return iou

    def compute_cofiring_proportions(self, cofiring_matrix: torch.Tensor,
                                   model_sums: Dict[str, torch.Tensor],
                                   energies: torch.Tensor,
                                   sort_by_energy: bool = True,
                                   eps: float = 1e-10) -> Dict[str, torch.Tensor]:
        """
        Compute proportion of fires that are cofires for each model.

        Returns:
            Dictionary mapping model names to proportion tensors
        """
        if sort_by_energy:
            concept_order = energies.argsort(descending=True)
        else:
            concept_order = torch.arange(energies.shape[0])

        sorted_cofire_sums = cofiring_matrix[:, concept_order].sum(dim=0)

        proportions = {}
        for model_name, sums in model_sums.items():
            proportions[model_name] = sorted_cofire_sums / (sums[concept_order] + eps)

        return proportions

    def compute_firing_entropy(self, model_sums: Dict[str, torch.Tensor],
                             union_fires: torch.Tensor, energies: torch.Tensor,
                             sort_by_energy: bool = True, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute entropy of firing patterns across models.

        Returns:
            Entropy values per concept
        """
        if sort_by_energy:
            concept_order = energies.argsort(descending=True)
        else:
            concept_order = torch.arange(energies.shape[0])

        # Compute probability distribution over models for each concept
        probabilities = []
        for model_name, sums in model_sums.items():
            model_probs = sums[concept_order] / (union_fires[concept_order] + eps)
            probabilities.append(model_probs)

        prob_dist = torch.stack(probabilities, dim=-1)  # [concepts, models]

        # Compute entropy: H = -sum(p * log2(p)), normalized by log M
        entropy = -torch.sum(prob_dist * torch.log2(prob_dist + eps), dim=1) / np.log2(prob_dist.shape[-1])

        return entropy

    def analyze(self, threshold: float = 0.1, sort_by_energy: bool = True) -> Tuple:
        """
        Main analysis function that computes all metrics.

        Args:
            threshold: Activation threshold for filtering
            sort_by_energy: Whether to sort concepts by energy

        Returns:
            Tuple containing (results, alignment_metrics, thresholded_activations,
                            union_fires, thresholded_sums, aggregated_energies)
        """
        print(f'Analysis settings: threshold={threshold}, sort_by_energy={sort_by_energy}')

        # Apply threshold and compute activation sums
        z_activations_thresh = self._apply_activation_threshold(threshold)
        activation_sums = self._compute_activation_sums(self.z_activations)
        activation_sums_thresh = self._compute_activation_sums(z_activations_thresh)

        # Aggregate energies across models
        energies_aggregated = torch.stack([
            self.results[name]['energies'] for name in self.results.keys()
        ], dim=-1).sum(dim=-1)

        # Compute union of fires across all models
        union_fires = torch.stack(list(activation_sums_thresh.values()), dim=0).sum(dim=0)

        # Compute basic statistics and update results
        stats = self._compute_activation_statistics(activation_sums, activation_sums_thresh)
        self._update_results_with_statistics(stats)

        # Compute alignment metrics
        cofiring_stats = self.compute_cofiring_statistics(z_activations_thresh)

        self.alignment_metrics = {
            'energies_agg': energies_aggregated,
            'union_fires': union_fires,
            'raw_cofiring': cofiring_stats,
            'cofire_iou': self.compute_cofire_iou_metric(
                cofiring_stats.matrix, union_fires, energies_aggregated, sort_by_energy
            ),
            'cofire_proportions': self.compute_cofiring_proportions(
                cofiring_stats.matrix, activation_sums_thresh, energies_aggregated, sort_by_energy
            ),
            'firing_entropy': self.compute_firing_entropy(
                activation_sums_thresh, union_fires, energies_aggregated, sort_by_energy
            )
        }

        #return (self.results, self.alignment_metrics, z_activations_thresh,
        #        union_fires, activation_sums_thresh, energies_aggregated)
        return self.alignment_metrics

    def _update_results_with_statistics(self, stats: Dict[str, ActivationStats]):
        """Update results dictionary with computed activation statistics."""
        for model_name, stat in stats.items():
            self.results[model_name].update({
                'Sums': stat.sums,
                'Sums_Thresh': stat.sums_thresh,
                'Total Fires': stat.total_fires,
                'Total Fires Above Thresh': stat.fires_above_threshold,
                'Total Dead': stat.dead_concepts_pct,
                'Total Dead After Thresh': stat.dead_concepts_thresh_pct
            })


def run_full_analysis_pipeline(config: Dict, config_viz: Dict, model_zoo: Dict) -> Dict:
    """
    Run the complete analysis pipeline for visualization demos.

    Args:
        config: General configuration
        config_viz: Visualization configuration (should specify small sample_size)
        model_zoo: Dictionary of model configurations (each with SAE pair)

    Returns:
        Dictionary containing all analysis results
    """
    print(f"Running analysis on {config_viz['sample_size']} samples for demo visualization")

    # Step 1: Extract activations
    print("=== Extracting Model Activations ===")
    extractor = ModelActivationExtractor(config, config_viz, model_zoo)
    model_activations, data_loader = extractor.extract_all_activations()

    # Step 2: Analyze SAE outputs
    print("=== Analyzing SAE Outputs ===")
    sae_analyzer = SAEAnalyzer(model_zoo, config, config_viz)
    sae_results = sae_analyzer.compute_sae_statistics(model_activations)

    # Step 3: Compute alignment metrics
    print("=== Computing Alignment Metrics ===")
    activation_analyzer = ActivationAnalyzer(sae_results, config, config_viz, model_zoo)

    final_results = activation_analyzer.analyze(
        threshold=0.1,
        sort_by_energy=True
    )

    return {
        'images': unwrap_dataloader(data_loader),
        'model_activations': model_activations,
        'sae_results': sae_results,
        'analysis_results': final_results
    }



def overlay_top_heatmaps_uni(
	images,
	heatmaps,
	concept_id,
	model_types=[],
	model_num=0,
	image_matches=None,
	num_images=10,
	image_save_path=None,
	cmap=None,
	alpha=0.35,
):
	"""
	*Adapted method above to visualize multiple models together, as well as original image and index info*
	Visualize the top activating image for a concepts and overlay the associated heatmap.
	This function sorts images based on the mean value of the heatmaps for a given concept and
	visualizes the top 10 images with their corresponding heatmaps.
	Parameters
	----------
	images : torch.Tensor or PIL.Image or np.ndarray
		Batch of input images of shape (batch_size, channels, height, width).
	z_heatmaps : torch.Tensor or np.ndarray
		Batch of heatmaps corresponding to the input images of
		shape (batch_size, height, width, num_concepts).
	concept_id : int
		Index of the concept to visualize.
	cmap : str, optional
		Colormap for the heatmap, by default 'jet'.
	alpha : float, optional
		Transparency of the heatmap overlay, by default 0.35.
	Returns
	-------
	None
	"""
	if heatmaps != None:
		if len(heatmaps) != len(image_matches):
			assert len(images) == len(heatmaps)
			assert heatmaps.shape[-1] > concept_id
			assert heatmaps.ndim == 4
	# if we handle the cmap, choose tab10 if number of concepts is less than 10
	# else choose a normal one
	if cmap is None:
		cmap = TAB10_ALPHA[concept_id] if heatmaps.shape[-1] < 10 else VIRIDIS_ALPHA
		# and enforce the alpha value to one, as the alpha is already handled by the colormap
		alpha = 1.0
	if image_matches == None:
		best_ids = _get_representative_ids(heatmaps, concept_id, num_images)
	else:
		best_ids = image_matches
	num_models = len(model_types)

	for i, idx in enumerate(best_ids):
		image = images[idx]
		width, height = get_image_dimensions(image)
		if heatmaps != None:
			if len(heatmaps) == len(images):
				heatmap = interpolate_cv2(heatmaps[idx, :, :, concept_id], (width, height))
			elif len(heatmaps) == len(image_matches):
				heatmap = interpolate_cv2(heatmaps[i, :, :], (width, height))
		if num_models > 1:
			plt.subplot(num_models+1, num_images, (model_num+1) * num_images + i + 1)
			if heatmaps == None:
				plt.gcf().text(
					0.1,
					1.13,
					f"n{i}_idx{idx}",
					transform=plt.gca().transAxes,
					fontsize=17,
					ha="left",
					va="center",
				)
			if i == len(best_ids) // 2 and heatmaps != None:
				plt.gcf().text(
					0.0,
					1.10,
					f"{model_types[model_num]}",
					transform=plt.gca().transAxes,
					fontsize=18,
					fontweight="medium",
					ha="center",
					va="center",
				)
		else:
			plt.subplot(2, 5, i + 1)

		show(image)

		if heatmaps != None:
			show(heatmap, cmap=cmap, alpha=alpha)
			if image_save_path != None:
				save_name = f"{image_save_path}/concept_id{concept_id}_imgidx{idx}"
				save_subplot_overlay(image, heatmap, cmap=cmap, alpha=alpha, save_path=save_name, model_name=model_types[model_num])


def save_subplot_overlay(image, heatmap, cmap="viridis", alpha=1.0, save_path=None, model_name=None):
	"""
	Save the overlay of an image and heatmap for the current subplot.

	Parameters:
			image (ndarray): Base image (H x W x 3 for RGB).
			heatmap (ndarray): Heatmap to overlay (H x W).
			cmap (str): Colormap for the heatmap (default: 'viridis').
			alpha (float): Alpha transparency for the heatmap (default: 0.6).
			save_path (str): Path to save the output (e.g., 'overlay_subplot.png').
	"""

	fig, ax = plt.subplots(figsize=(1, 1), dpi=224)
	plt.axis("off")
	ax.set_position([0, 0, 1, 1])
	image = np_channel_last(image)
	image = normalize(image)
	heatmap = np_channel_last(heatmap)
	heatmap = normalize(heatmap)
	plt.imshow(image, aspect="auto") # a bit redundant but 'ok' for now
	fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0)

	plt.imshow(heatmap, cmap=cmap, alpha=alpha, aspect="auto")
	fig.savefig(f"{save_path}_{model_name}.png", bbox_inches="tight", pad_inches=0.0)

	plt.close(fig)
