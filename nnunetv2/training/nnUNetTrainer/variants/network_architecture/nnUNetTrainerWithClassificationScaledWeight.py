import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerWithClassification import nnUNetTrainerWithClassification

class nnUNetTrainerWithClassificationScaledWeight(nnUNetTrainerWithClassification):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device) 
        self.cls_loss_weight = 1
        self.patience = 30