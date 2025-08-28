"""
Sparse Autoencoder (SAE) module of Overcomplete.
"""

from .base import SAE, test_import_func
from .dictionary import DictionaryLayer
from .optimizer import CosineScheduler
from .losses import mse_l1
from .modules import MLPEncoder, AttentionEncoder, ResNetEncoder
from .factory import EncoderFactory
from .jump_sae import JumpSAE, jump_relu, heaviside
from .topk_sae import TopKSAE
from .qsae import QSAE
