![Overcomplete USAE Architecture](https://yorkucvil.github.io/UniversalSAE/img/fig1.png)

# Universal Sparse Autoencoders

[Harrish Thasarathan](https://harry-thasarathan.github.io),
[Julian Forsyth](https://j-forsyth.github.io),
[Thomas Fel](https://thomasfel.fr/),
[Matthew Kowal](https://mkowal2.github.io/),
[Konstantinos G. Derpanis](https://csprofkgd.github.io/)

Universal Sparse Autoencoders (USAEs) create a universal, interpretable concept space that reveals what multiple vision models learn in common about the visual world.

- **[Paper (Arxiv)](https://arxiv.org/abs/2502.03714)**
- **[Project Page](https://yorkucvil.github.io/UniversalSAE/)**
- **[Interactive Demo: Concept Explorer](https://yorkucvil.github.io/UniversalSAE/demo)**

## Getting Started

See setup instructions for [Overcomplete](https://github.com/KempnerInstitute/overcomplete). Python 3.8 or later is required.

Run **_uni_demo.py_** to set up and train a USAE: ```python -m uni_demo```

## Features

- **USAE training framework**
- **Evaluation Notebook** to validate metrics of trained USAE

To replicate the model used for our paper, see config.yaml file for hyperparameters. The training script assumes activations are in specified form (see uni_demo.py for details). Our dataset/loader is customized for Imagenet and assumes activations have been cached to .npz form.

## Data Pipeline Details
During **training**, each batch is expected to be in the form:
- Model Activations X and labels Y in tuple ( X{Dict: (models s_1...s_k)}, Y)
where each value from model key s_i is in form N(batch size) x d_i(activation dimension)

The **Activation Dataset/Loader** is customized for training on Imagenet:
- requires model activations to be cached
- assumes activation directory follows format of Imagenet directory
- recommended to store all models' activations for each image in one npz file (hence 'combined_npz' param)
- see models.py for model activation details

To **cache model activations from Imagenet**, see scripts in the caching_acts folder.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{
thasarathan2025universal,
title={Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment},
author={Harrish Thasarathan and Julian Forsyth and Thomas Fel and Matthew Kowal and Konstantinos G. Derpanis},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=UoaxRN88oR}
}
```

## Acknowledgements

This project was built on top of Thomas Fel's [Overcomplete](https://github.com/KempnerInstitute/overcomplete) library.
