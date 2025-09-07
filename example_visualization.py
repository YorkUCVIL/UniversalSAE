from universal_sae.uni_analysis import *
from overcomplete.config.config_loader import load_model_zoo
from overcomplete.models import ViT, SigLIP, DinoV2
from overcomplete.sae import TopKSAE
from overcomplete.visualization.top_concepts import _get_representative_ids

from universal_sae.uni_analysis import interpolate_patch_tokens
from tqdm import tqdm
from typing import Dict, Tuple
from einops import rearrange

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_cofiring_importance(analysis, save_dir):
    raw_cofires = analysis['raw_cofiring'].matrix.sum(dim=0)

    union_thresh_mask = raw_cofires > 0
    #union_thresh = analysis['union_fires'][union_thresh_mask]

    iou_values_thresh = raw_cofires[union_thresh_mask]
    energies_agg_thresh = analysis['energies_agg'][union_thresh_mask] / 3
    mask = (iou_values_thresh > 0) & (energies_agg_thresh > 0)
    iou_values_filtered = iou_values_thresh[mask]
    importance_values_filtered = energies_agg_thresh[mask]

    plt.figure(figsize=(10, 6))

    # Plot all points that are positive
    plt.scatter(iou_values_filtered, importance_values_filtered, alpha=0.5)

    # Set both axes to log scale
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Number of Co-Fires (log scale)')
    plt.ylabel('Concept Importance (log scale)')

    # Fit only on valid (positive) values
    z = np.polyfit(np.log10(iou_values_filtered), np.log10(importance_values_filtered), 1)
    x_log = np.linspace(np.log10(min(iou_values_filtered)), np.log10(max(iou_values_filtered)), 100)
    y_log = z[0] * x_log + z[1]
    plt.plot(10**x_log, 10**y_log, "r--", alpha=0.8)

    # Calculate correlation on filtered log values
    correlation = np.corrcoef(np.log10(iou_values_filtered), np.log10(importance_values_filtered))[0,1]
    plt.title(f'Correlation between Cofiring and Importance (log-log r={correlation:.3f})')

    plt.savefig(f"{save_dir}/cofiring_vs_importance.png")

def plot_entropy(firing_entropy, save_dir):
    #firing entropy histogram
    firing_entropy = analysis['firing_entropy']
    print(firing_entropy.shape)
    print(firing_entropy.max(), firing_entropy.min())


    plt.figure(figsize=(5, 4.5))  # Standard column width for ICML

    # Assuming entropy is your vector of entropy values
    plt.hist(firing_entropy.cpu().numpy(), bins=30, color='#2878B5', alpha=0.8,
             edgecolor='black', linewidth=0.8)

    # Add grid with light gray lines behind the data
    plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # Labels with LaTeX formatting
    plt.xlabel('Normalized Firing Entropy ($H_k$)')
    plt.ylabel('Number of Concepts')

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Optional: Add vertical line at maximum entropy (log2(3) for 3 models)
    max_entropy = 1.0  # Since it's normalized
    plt.axvline(x=max_entropy, color='#C14B4B', linestyle='--', alpha=0.8,
                label='Maximum Entropy')
    plt.legend(frameon=False)
    plt.title(f'Concept Firing Entropy (Normalized)')
    plt.savefig(f"{save_dir}/firing_entropy.png")

if __name__ == '__main__':

    models_used = {
            "ViT": ViT,
            "DinoV2": DinoV2,
            "SigLIP": SigLIP,
        }

    #model from our paper
    #model_config_path = 'checkpoints/config_dino_siglip_vit.yaml'

    #model you trained
    model_config_path = 'results/config.yaml'

    CONFIG, CONFIG_viz, sae_params, model_zoo = load_model_zoo(model_config_path, models_used)

    for model in model_zoo:
        sae = TopKSAE(input_shape=model_zoo[model]['input_shape'], n_components=CONFIG['nb_components'], device='cuda', **sae_params)
        print(model_zoo[model]['checkpoint_path'])
        checkpoint = torch.load(model_zoo[model]['checkpoint_path'])
        sae.load_state_dict(checkpoint['model_state_dict'])
        model_zoo[model]['sae'] = sae.to('cuda').eval()

    extractor = ModelActivationExtractor(CONFIG, CONFIG_viz, model_zoo)
    model_activations, data_loader = extractor.extract_all_activations()

    results = run_full_analysis_pipeline(CONFIG, CONFIG_viz, model_zoo)
    analysis = results['analysis_results']

    save_dir = os.path.join(os.getcwd(), "results")
    heatmaps_dir = os.path.join(save_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    #example cofiring vs importance/entropy plot from paper
    plot_cofiring_importance(analysis, save_dir)
    plot_entropy(analysis['firing_entropy'], save_dir)

    #Heatmap visualization of top 15 concepts by energy for the same val imagenet indices, all model visualized together in a single image
    important_concepts = analysis['energies_agg'].argsort().flip(dims=(0,))
    images = results['images']

    for rank, concept_id in enumerate(important_concepts[:15]):
        fig = plt.figure(figsize=(25, len(list(model_zoo.keys())) * 3))

        avg_heatmaps = torch.stack([results['sae_results'][m]["heatmaps"] for m in list(model_zoo.keys())], dim=0).mean(dim=0)
        image_matches = _get_representative_ids(avg_heatmaps, concept_id, 10)

        for model_num, model_type in enumerate(list(model_zoo.keys())):
            overlay_top_heatmaps_uni(
                images,
                results['sae_results'][model_type]['heatmaps'],
                concept_id,
                model_types=list(model_zoo.keys()),
                model_num=model_num,
                image_matches=image_matches,
                num_images=10,
                image_save_path=None
            )

        # spacing adjustments
        plt.tight_layout(h_pad=1.0, w_pad=0.5)
        plt.subplots_adjust(
            top=0.90,
            left=0.08,
            right=0.98,
            hspace=0.3
        )

        plt.savefig(f"{heatmaps_dir}/rank{rank}_index{concept_id}.png",
                    bbox_inches='tight', dpi=150)
        plt.show()
        plt.close()