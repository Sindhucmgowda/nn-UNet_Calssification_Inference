from typing import Union, Tuple, List
import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerClsFeatureScaling import nnUNetTrainerClsFeatureScaling 

class nnUNetTrainerClsFeatureScaledWeight(nnUNetTrainerClsFeatureScaling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device) 
        self.cls_loss_weight = 100
        self.patience = 80