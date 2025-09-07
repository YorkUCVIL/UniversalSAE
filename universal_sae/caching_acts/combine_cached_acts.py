"""
Script to combine separate cached activations from respective models into combined npz files for each image
eg. some matching activation files, which are stored in the same directory...
.../train/n15075141/...  n15075141_9993_DinoV2.npz, n15075141_9993_SigLIP.npz, etc. --> n15075141_9993_combined.npz
"""

import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def process_class_directory(args):
    """Process a single class directory"""
    class_dir, split_dir, output_split_dir, sources = args

    class_path = os.path.join(split_dir, class_dir)
    if not os.path.isdir(class_path):
        return

    # Create output directory
    output_class_dir = os.path.join(output_split_dir, class_dir)
    os.makedirs(output_class_dir, exist_ok=True)

    # Get all activation files for first source
    source_files = glob.glob(os.path.join(class_path, f'*_{sources[0]}.npz'))

    for source_file in source_files:
        base_name = os.path.basename(source_file).replace(f'_{sources[0]}.npz', '')

        combined_acts = {}
        for source in sources:
            act_path = os.path.join(class_path, f'{base_name}_{source}.npz')
            if not os.path.exists(act_path):
                print(f"Missing activation file: {act_path}")
                continue

            act = np.load(act_path)['activation']
            combined_acts[source] = act

        output_path = os.path.join(output_class_dir, f'{base_name}_combined.npz')
        np.savez(output_path, **combined_acts)

def combine_activations(activation_root: str, output_root: str, split: str, sources: list, num_workers: int = 8):
    split_dir = os.path.join(activation_root, split)
    output_split_dir = os.path.join(output_root, split)

    class_dirs = [d for d in os.listdir(split_dir)
                 if os.path.isdir(os.path.join(split_dir, d))]

    args_list = [(class_dir, split_dir, output_split_dir, sources)
                 for class_dir in class_dirs]

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap(process_class_directory, args_list),
            total=len(class_dirs),
            desc="Combining activations"
        ))



if __name__ == '__main__':
    #If you have enough GPU space to load all models you can save activations in a single npz from the start (cache_imagenet_activations.py)

    activation_root = "" # folder containing separate npz files for each model's activations
    output_root = "" # output folder for combined_npz activations
    split = 'train'
    sources = [] # list of model names eg. ["SigLIP ..."

    combine_activations(activation_root=activation_root,
                        output_root = output_root,
                        split=split,
                        sources=sources,
                        num_workers=12)