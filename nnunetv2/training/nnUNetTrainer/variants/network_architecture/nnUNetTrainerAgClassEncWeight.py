from typing import Union, Tuple, List
import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerAgClassEncOld import nnUNetTrainerAgClassEnc

class nnUNetTrainerAgClassEncWeight(nnUNetTrainerAgClassEnc):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cls_loss_weight = 0.5
        self.patience = 80

