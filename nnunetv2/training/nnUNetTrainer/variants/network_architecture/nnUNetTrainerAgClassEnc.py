from typing import Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerWithClassification import nnUNetTrainerWithClassification
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerClsFeatureScaling import NetworkWithFeatureScaling, NetworkWithClassification

class normGradientOp(torch.autograd.Function):
    """
    Custom pytorch function that copies the input on forward and normalizes 
    the gradient to norm 1 on backward

    Inspired by cloneofsimo on X
    """
    @staticmethod
    def forward(ctx, input, norm_value):
        ctx.save_for_backward(input)
        ctx.norm_value = norm_value
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        norm_value = ctx.norm_value
        
        grad_input = grad_output.clone()
        grad_input_norm = torch.norm(grad_input)

        if grad_input_norm == 0:
            grad_input_normed = grad_input
        else:
            grad_input_normed =  grad_input *  (norm_value / grad_input_norm)

        return grad_input_normed, None

class AttentionGate(nn.Module):
    def __init__(self, in_channels, leaky_relu_slope=0.01):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, f_enc, f_cls):
        # f_enc, f_cls: (B, C, D, H, W)
        # Ensure spatial sizes match -- in the first pass
        # if f_enc.shape[2:] != f_cls.shape[2:]:
        #     # f_enc = F.interpolate(f_enc, size=f_cls.shape[2:], mode='trilinear', align_corners=False)
        #     f_enc = F.interpolate(f_enc, size=f_cls.shape[2:], mode='trilinear', align_corners=False)

        sum_feat = F.leaky_relu(f_enc + f_cls, negative_slope=self.leaky_relu_slope)
        feat = self.conv(sum_feat)  # (B, 1, D, H, W)
        alpha = torch.sigmoid(feat)  # (B, 1, D, H, W)

        return f_cls * alpha


class ClassificationEncoderWithAttention(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        for idx in range(len(encoder_channels)):
            in_ch = encoder_channels[idx-1] if idx > 0 else encoder_channels[0]
            out_ch = encoder_channels[idx]
            self.stages.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=2))
            self.norms.append(nn.InstanceNorm3d(out_ch, affine=True))
            
            self.attention_gates.append(AttentionGate(out_ch))

    def forward(self, seg_encoder_features):
        # seg_encoder_features: list of features from segmentation encoder, deepest last
        cls_features = []
        f_cls = seg_encoder_features[0]  # Start from the first encoder feature

        for i, (stage, norm, ag) in enumerate(zip(self.stages, self.norms, self.attention_gates)):
            # print(f"Stage {i}: x shape before stage: {f_cls.shape}, expected in_ch: {stage.in_channels}")
            f_cls = stage(f_cls)
            f_cls = norm(f_cls)
            f_cls = F.leaky_relu(f_cls, negative_slope=0.01)
            # print(f"Stage {i}: x shape after stage: {f_cls.shape}")
            f_enc = seg_encoder_features[i]           
            # print(f'expected f_enc shape: {f_enc.shape}')

            if f_cls.shape[2:] != seg_encoder_features[i].shape[2:]:
                # Just in case, interpolate to match exactly
                f_cls = F.interpolate(f_cls, size=seg_encoder_features[i].shape[2:], mode='trilinear', align_corners=False)

            f_cls = ag(f_enc, f_cls)
            cls_features.append(f_cls)
            
        return cls_features

class NetworkWithAgClassEnc(NetworkWithFeatureScaling):
    def __init__(self, base_network, num_classes=3):
        super().__init__(base_network, num_classes)
        # calls add_classification_head() in the base class

        encoder_channels = [s.output_channels for s in self.encoder.stages]
        self.classification_encoder = ClassificationEncoderWithAttention(encoder_channels)
        self.norm_gradient_op = normGradientOp.apply

        self._encoder_features = []
        self._hooks = []

        # Add hooks to the encoder stages
        for stage in self.encoder.stages:
            self._hooks.append(stage.register_forward_hook(
                self.feature_extraction_hook_fn()
        ))

    def feature_extraction_hook_fn(self):
        def hook(module, input, output):
            self._encoder_features.append(output)
        return hook

    def add_classification_head(self):
        # Use the deepest classification encoder feature for the head
        deepest_channels = self.encoder.stages[-1].output_channels
        self.classification_head = nn.Sequential(
            nn.Conv3d(deepest_channels, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.InstanceNorm3d(256, affine=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.InstanceNorm3d(256, affine=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x, seg_target=None):
        del self._encoder_features
        self._encoder_features = []

        # Get segmentation output
        seg_output = self.base_network(x)
        # seg_encoder_features = [s.output for s in self.encoder.stages]  # Assume each stage stores its output
        seg_encoder_features = self._encoder_features
        # Classification encoder with attention gates
        cls_features = self.classification_encoder(seg_encoder_features)
        
        # Use the deepest feature for classification head
        cls_features = cls_features[-1]
        lesion_seg = self.get_seg_by_class(seg_output, cls_features.shape[2:],
                                         seg_target=seg_target,
                                         class_index=2)

        cls_features = lesion_seg.detach() * cls_features

        cls_output = self.classification_head(cls_features)
        # Wrap outputs as in the base class

        seg_output = self.norm_gradient_op(seg_output, 0.1)
        cls_output = self.norm_gradient_op(cls_output, 0.1)

        wrapped_output = self.wrap_network_outputs(seg_output, cls_output)
                
        return wrapped_output

class nnUNetTrainerAgClassEnc(nnUNetTrainerWithClassification):
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
            
        return NetworkWithAgClassEnc(base_network, num_classes=3)