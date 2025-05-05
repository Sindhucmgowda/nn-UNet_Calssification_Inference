from typing import Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerWithClassification import nnUNetTrainerWithClassification
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerWithClassification import NetworkWithClassification

class NetworkWithFeatureScaling(NetworkWithClassification):
    def __init__(self, base_network, num_classes=3):
        super().__init__(base_network, num_classes)

    def add_classification_head(self):
        # This one uses the last decoder stage instead
        # cls_feature_module = self.decoder.stages[-1]
        cls_feature_module = self.encoder.stages[-1]
        cls_feature_channels = cls_feature_module.output_channels

        self.classification_head = nn.Sequential(
            nn.Conv3d(cls_feature_channels, 256, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self.feature_extraction_hook = cls_feature_module.register_forward_hook(
            self.feature_extraction_hook_fn()
        )

    def get_seg_by_class(self, seg_output, output_shape, 
                       seg_target=None,
                       normalize=True,
                       add_gaussian_noise=True,
                       class_index=2):
        '''
        Get the segmentation for a specific class resampled to the output shape
        
        seg_output: output of the network
        output_shape: nn.interpolate() to this shape
        seg_target: ground truth segmentation
        normalize: whether to normalize the segmentation
        add_gaussian_noise: whether to add gaussian noise 0.03 * torch.randn_like(seg) to the segmentation
                            only used if seg_target is not None
        class_index: index of the class to get the segmentation for
        '''


        def _normalize(seg):
            orig_shape = seg.shape
            seg = seg.view(seg.shape[0], -1)
            seg = seg - torch.min(seg, dim=1, keepdim=True).values
            seg = seg / (torch.max(seg, dim=1, keepdim=True).values + 1e-6)
            seg = seg.view(orig_shape)
            
            return seg

        if seg_target is not None:
            # Only used in train_step
            # use ground truth lesion seg
            if isinstance(seg_target, list):
                seg = (seg_target[0] == class_index).float()
            else:
                seg = (seg_target == class_index).float()

            # add gaussian noise to lesion seg
            if add_gaussian_noise:
                seg = seg + torch.randn_like(seg) * 0.03

        else:
            # get predicted lesion seg probabilities and normalize to [0, 1]
            if isinstance(seg_output, list):
                # If deep supervision is enabled, seg_output is a list of tensors
                seg = F.softmax(seg_output[0], dim=1)[:, class_index:class_index+1, ...] # [B, 1, D, H, W]
            else:
                seg = F.softmax(seg_output, dim=1)[:, class_index:class_index+1, ...] # [B, 1, D, H, W]
    
        # If we use the highest resolution features for classification, 
        # they are the same size as the target and don't need to be resized
        
        if seg.shape[2:] != output_shape:
            # trilinear resample lesion seg to the same size as self.intermediate_features
            seg = F.interpolate(
                seg, 
                size=output_shape, 
                mode='trilinear', 
                align_corners=False
            )

        if normalize:
            seg = _normalize(seg)

        return seg

    def forward(self, x, seg_target=None):
        # Get segmentation output and populate self.intermediate_features
        seg_output = self.base_network(x)
        lesion_seg = self.get_seg_by_class(seg_output, 
                                 self.intermediate_features.shape[2:],
                                 seg_target=seg_target,
                                 class_index=2)

        # Only use features near the lesion
        # Uses gaussian smoothed GT during training and predicted lesion seg during inference
        combined_features = lesion_seg.detach() * self.intermediate_features
        # combined_features = self.intermediate_features + lesion_seg * self.intermediate_features
        
        # Pass through classification head
        cls_output = self.classification_head(combined_features)

        # Wrap segmentation and classification outputs
        wrapped_output = self.wrap_network_outputs(seg_output, cls_output)

        return wrapped_output
    
class nnUNetTrainerClsFeatureScaling(nnUNetTrainerWithClassification):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device) 

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                 arch_init_kwargs: dict,
                                 arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                 num_input_channels: int,
                                 num_output_channels: int,
                                 enable_deep_supervision: bool = True) -> nn.Module:
        # First get the base U-Net architecture
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )
            
        return NetworkWithFeatureScaling(base_network, num_classes=3)