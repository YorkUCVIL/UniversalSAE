"""
Cache forward feature pass of various models on Imagenet dataset (train set).
    - could be modified to cache multiple models at once.
"""

import sys; sys.path.insert(0, '..')
from overcomplete.models import ViT, DinoV2, SigLIP
from torchvision.datasets import ImageNet
import torch
import numpy as np
import os
from tqdm import tqdm
import gzip

# Save a tensor with gzip compression
def save_tensor_compressed(tensors, path):
    with gzip.open(path, 'wb') as f:
        torch.save(tensors, f)

# Load a compressed tensor
def load_tensor_compressed(path):
    with gzip.open(path, 'rb') as f:
        tensor = torch.load(f)
    return tensor

# Save tensor in NPY format with compression
def save_tensor_npz(tensors, path):
    np.savez_compressed(path, activation=tensors[0].numpy(), label=tensors[1].numpy())

# Load tensor from NPZ format
def load_tensor_npz(path):
    data = np.load(path)
    activation = torch.tensor(data['activation'])
    label = torch.tensor(data['label'])
    return activation, label

if __name__ == '__main__':

    model = ViT().cuda() # Bx197x768
    # model = SigLIP().cuda() # Bx196x768 (no cls token)
    # model = DinoV2().cuda() #Bx(256[16x16tokens]+1cls+0register=257)x384

    model.eval()

    print("Caching model activations: ", model.__class__.__name__)
    print("Loading ImageNet")

    imagenet_data = ImageNet("[directory]", transform=model.preprocess)

    print("ImageNet Loaded")

    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=512,
                                              shuffle=False,
                                              num_workers=8)

    # Path to cache activations in the ImageNet directory structure
    path_to_cache = "Insert Here"

    print(model.__class__.__name__, path_to_cache)

    # Iterate through the dataloader
    for i, batch in enumerate(tqdm(data_loader)):
        x, y = batch
        x = x.cuda()
        y = y.cpu()

        # Retrieve batch filenames from the dataset
        image_paths = data_loader.dataset.samples[i * data_loader.batch_size:(i + 1) * data_loader.batch_size]

        # ViT
        output_features = model.model.forward_features(x) # to avoid removing cls token

        # SigLIP -> No cls token in forward features of vit_base_patch16_siglip_224
        # output_features = model.forward_features(x).cpu()

        # DinoV2
        # output_dict = model.model.forward_features(x)
        # cls = output_dict['x_norm_clstoken'].unsqueeze(1)
        # output_features = torch.cat((cls, output_dict['x_norm_patchtokens']), dim=1)

        # Loop through each sample in the batch
        for j in range(x.size(0)):  # Loop through batch size

            # Get the corresponding image path (original ImageNet file path)
            image_path = image_paths[j][0]  # Assuming image path is in the 0th index

            # Convert ImageNet image path to the corresponding cache path
            relative_path = os.path.relpath(image_path, data_loader.dataset.root)
            activation_path = os.path.join(path_to_cache, relative_path)

            # Get only the directory part, excluding the image filename
            activation_dir = os.path.dirname(activation_path)

            # Ensure the directory exists, creating it if necessary
            os.makedirs(activation_dir, exist_ok=True)

            # Save both the activations and label
            activation_filename = activation_path.replace('.JPEG', f"{model.__class__.__name__}")  # change extension

            '''save as float 16 not 32'''
            save_tensor_npz(path=activation_filename, tensors=(output_features[j].detach().clone(), y[j].detach().clone()))