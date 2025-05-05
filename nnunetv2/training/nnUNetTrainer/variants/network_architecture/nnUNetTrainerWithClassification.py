from typing import Union, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch.cuda.amp import autocast, GradScaler
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.logging.nnunet_logger_with_classification import nnUNetLoggerWithClassification
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from acvl_utils.cropping_and_padding.padding import pad_nd_image
import os 
import tqdm
from nnunetv2.utilities.metric_handling.metricreloaded_metrics import macro_f1_score, dsc, micro_dice_score
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save, export_cls_prediction_from_logits
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, nnUNetPredictorWithClassification
import warnings
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
import multiprocessing
import time 
from time import sleep

class NetworkWithClassification(nn.Module):
    def __init__(self, base_network, num_classes=3):
        super().__init__()
        self.base_network = base_network
        # Number of global classification classes to predict
        self.num_classes = num_classes
        self.encoder = self.base_network.encoder
        self.decoder = self.base_network.decoder

        self.add_classification_head()

    def add_classification_head(self):
        # Feature to use for classification
        cls_feature_module = self.encoder.stages[-1]
        cls_feature_channels = cls_feature_module.output_channels

        self.classification_head = nn.Sequential(
            nn.Conv3d(cls_feature_channels, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self.feature_extraction_hook = cls_feature_module.register_forward_hook(
            self.feature_extraction_hook_fn()
        )

    def feature_extraction_hook_fn(self):
        def hook(module, input, output):
            self.intermediate_features = output
            # self.feature_layer_name = module.__class__.__name__
            # print(f"Feature extraction hook capturing features from layer: {self.feature_layer_name}")
        return hook

    def wrap_network_outputs(self, seg_output, cls_output):
        ''' 
        Wraps the segmentation output with the classification output.
        Classification output is broadcasted to the same shape as seg_output[0] or seg_output and concatenated.
        
        If deep supervision is enabled, the seg_output is a list of tensors.
        If deep supervision is not enabled, the seg_output is a single tensor.
        '''
        # Broadcast cls_output to the same shape as seg_output[0] or seg_output
        cls_output = cls_output.view(cls_output.size(0), cls_output.size(1), 1, 1, 1) 

        if isinstance(seg_output, list):
            # for deep supervision, the seg_output is a list of tensors
            expand_shape = list(seg_output[0].shape)
            expand_shape[1] = -1
            cls_output = cls_output.expand(*expand_shape)
            seg_output[0] = torch.cat((seg_output[0], cls_output), dim=1)
        else:
            expand_shape = list(seg_output.shape)
            expand_shape[1] = -1
            cls_output = cls_output.expand(*expand_shape)
            seg_output = torch.cat((seg_output, cls_output), dim=1)
        
        return seg_output

    def unwrap_network_outputs(self, output, mean_cls=False):
        '''
        Unwraps the segmentation output with the classification output.
        If deep supervision is enabled, the output is a list of tensors.
        If deep supervision is not enabled, the output is a single tensor.

        :param output: output from the network. shape: (B, C, D, H, W)

        :return: seg_output is a list of tensors if self.enable_deep_supervision is True
        :return: seg_output is a single tensor if self.enable_deep_supervision is False
        :return: cls_output is a single tensor
        '''
        
        if isinstance(output, list):
            # If deep supervision is enabled, the output is a list of tensors
            cls_output = output[0][:, -self.num_classes:, :, :, :]
            output[0] = output[0][:, :-self.num_classes, :, :, :]
        else:
            cls_output = output[:, -self.num_classes:, :, :, :]
            output = output[:, :-self.num_classes, :, :, :]
        
        # If mean_cls is True, take the mean of the cls_output over the spatial dimensions or just take one element
        if mean_cls:
            cls_output = torch.mean(cls_output, dim=(2,3,4))
        else:
            cls_output = cls_output[:, :, 0, 0, 0]

        seg_output = output

        return seg_output, cls_output


    def forward(self, x):
        # Get segmentation output
        seg_output = self.base_network(x) 
        cls_output = self.classification_head(self.intermediate_features) 

        # wrap segmentation and classification outputs
        wrapped_output = self.wrap_network_outputs(seg_output, cls_output)

        return wrapped_output
    
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    Original: https://github.com/AdeelH/pytorch-multi-class-focal-loss
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class nnUNetTrainerWithClassification(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device) 
        self.num_classes = 3
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        # Add classification loss
        self.classification_loss = FocalLoss(gamma=2)
        self.save_every = 2
        
        # For batch size 2
        self.num_epochs = 2
        # self.num_iterations_per_epoch = 126
        # self.num_val_iterations_per_epoch = 18

        self.num_iterations_per_epoch = 1
        self.num_val_iterations_per_epoch = 11
        
        self.logger = nnUNetLoggerWithClassification()
        self.cls_loss_weight = 1
        
        # Early stopping parameters
        self.patience = 10  # Number of epochs to wait for improvement
        self.min_delta = 1e-4  # Minimum change to qualify as an improvement

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

        # TODO: num_classes hardcoded for now
        return NetworkWithClassification(base_network, num_classes=3)

    def train_step(self, batch: dict) -> dict:
        # Get data and targets
        data = batch['data']
        target = batch['target']
        identifiers = batch['keys']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Get classification target
        cls_target = torch.tensor([int(identifier.split('_')[2]) for identifier in identifiers])
        cls_target = cls_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # import pdb; pdb.set_trace()

        with torch.autocast(device_type=self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Forward pass
            combined_output = self.network(data, target)
            seg_output, cls_output = self.network.unwrap_network_outputs(combined_output)

            # Calculate losses
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.classification_loss(cls_output, cls_target)
            
            # Combined loss
            total_loss = seg_loss + self.cls_loss_weight * cls_loss  # Adjust weight as needed
            # total_loss = seg_loss   # Adjust weight as needed
            # total_loss = cls_loss  # Adjust weight as needed

            pred = torch.argmax(cls_output, dim=1)
            cls_acc = (pred == cls_target).float().mean()
    
        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': total_loss.detach().cpu().numpy(), 
                'seg_loss': seg_loss.detach().cpu().numpy(), 
                'cls_loss': cls_loss.detach().cpu().numpy(),
                'cls_acc': cls_acc.detach().cpu().numpy()}

    # Print gradient norms for encoder, segmentation head, and classification head
    def debug_check_gradients(self):
        seg_grads = {}
        cls_grads = {}
        encoder_grads = {}

        # Check encoder gradients
        for name, param in self.network.encoder.named_parameters():
            if param.grad is not None:
                encoder_grads[name] = {
                    'grad_norm': param.grad.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item()
                }
                print(f"Encoder layer {name} - Grad norm: {encoder_grads[name]['grad_norm']:.6f}, Mean: {encoder_grads[name]['grad_mean']:.6f}, Std: {encoder_grads[name]['grad_std']:.6f}")

        # Check segmentation head gradients
        for name, param in self.network.decoder.named_parameters():
            if param.grad is not None:
                seg_grads[name] = {
                    'grad_norm': param.grad.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item()
                }
                print(f"Decoder layer {name} - Grad norm: {seg_grads[name]['grad_norm']:.6f}, Mean: {seg_grads[name]['grad_mean']:.6f}, Std: {seg_grads[name]['grad_std']:.6f}")
        
        # Check classification head gradients
        for name, param in self.network.classification_head.named_parameters():
            if param.grad is not None:
                cls_grads[name] = {
                    'grad_norm': param.grad.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item()
                }
                print(f"Classification layer {name} - Grad norm: {cls_grads[name]['grad_norm']:.6f}, Mean: {cls_grads[name]['grad_mean']:.6f}, Std: {cls_grads[name]['grad_std']:.6f}")

        # Print summary of gradient flow
        print("\nGradient Flow Summary:")
        # print(f"Classification Loss: {cls_loss.item():.6f}")
        print(f"Encoder gradient norms: {[v['grad_norm'] for v in encoder_grads.values()]}")
        print(f"Segmentation head gradient norms: {[v['grad_norm'] for v in seg_grads.values()]}")
        # print(f"Segmentation layers:  {seg_grads.keys()}")
        print(f"Classification head gradient norms: {[v['grad_norm'] for v in cls_grads.values()]}")
        # print(f"Classification layers:  {cls_grads.keys()}")

    def validation_step(self, batch: dict) -> dict:
        # Get data and targets
        data = batch['data']
        target = batch['target']
        identifiers = batch['keys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        class_labels = torch.tensor([int(identifier.split('_')[2]) for identifier in identifiers]) 
        class_labels = class_labels.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(device_type=self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            combined_output = self.network(data)
            seg_output, cls_output = self.network.unwrap_network_outputs(combined_output)
            del data
            # l = self.loss(output, target)
        
            seg_loss = self.loss(seg_output, target)
            class_loss = self.classification_loss(cls_output, class_labels)
            total_loss = seg_loss + self.cls_loss_weight * class_loss

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, seg_output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            # no need for softmax
            seg_output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, seg_output_seg, 1)
            # del seg_output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        
        # Calculate classification metrics
        class_pred = torch.argmax(cls_output, dim=1)
        class_acc = (class_pred == class_labels).float().mean()
        class_total = len(class_labels)
        # return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        return {
        'loss': total_loss.detach().cpu().numpy(),
        'seg_loss': seg_loss.detach().cpu().numpy(),
        'class_loss': class_loss.detach().cpu().numpy(),
        'tp_hard': tp_hard,
        'fp_hard': fp_hard,
        'fn_hard': fn_hard,
        'class_acc': class_acc.detach().cpu().numpy(),
        'class_total': class_total,
        'class_pred': class_pred.detach().cpu().numpy(),
        'class_labels': class_labels.detach().cpu().numpy(), 
        'seg_output_seg': seg_output_seg.detach().cpu().numpy(),
        'target': target.detach().cpu().numpy()
        }

    def lesion_dsc(self, pred, ref):
        """
        Calculates the Dice score specifically for pancreas lesions (label=2).
        pred and ref are numpy arrays
        Returns:
            float: Dice score for pancreas lesions
        """
        # Create binary masks for lesions (label == 2)

        pred_binary = np.where(pred == 2, 1, 0)
        ref_binary = np.where(ref == 2, 1, 0)
        
        # Calculate Dice score using BinaryPairwiseMeasures
        return dsc(pred_binary, ref_binary)

    def whole_pancreas_dsc(self, pred, ref):
        """
        Calculates the Dice score for the whole pancreas, combining normal pancreas (label=1)
        and pancreas lesion (label=2) into a single binary mask.
    
        Returns:
        float: Dice score for the whole pancreas
        """    # Create binary masks for whole pancreas (label > 0)

        pred_binary = np.where(pred > 0, 1, 0)
        ref_binary = np.where(ref > 0, 1, 0)
        
        return dsc(pred_binary, ref_binary)


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        
        # Handle segmentation metrics (original nnUNet logic)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()
            
            # Gather segmentation metrics
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            # Gather classification metrics
            # Gather losses
            class_accuracy = [None for _ in range(world_size)]
            dist.all_gather_object(class_accuracy, outputs_collated['class_acc'])
            class_acc = np.vstack(class_accuracy).mean()

            # Gather losses
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()


        loss_here = np.mean(outputs_collated['loss'])
        
        # classification metrics 
        class_accuracy = np.mean(outputs_collated['class_acc'])

        class_pred = outputs_collated['class_pred']
        class_labels = outputs_collated['class_labels']
        seg_output_seg = outputs_collated['seg_output_seg']
        target = outputs_collated['target']

        # Calculate and log segmentation metrics
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        macro_f1 = macro_f1_score(class_pred, class_labels, [0,1,2]) 
        micro_dscore = micro_dice_score(seg_output_seg, target, [0,1,2])
        whole_pan_dsc = self.whole_pancreas_dsc(seg_output_seg, target)
        les_dsc = self.lesion_dsc(seg_output_seg, target)
        
         # Log all metrics
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('class_accuracy', class_accuracy, self.current_epoch)
        self.logger.log('macro_f1', macro_f1, self.current_epoch)
        self.logger.log('whole_pancreas_dsc', whole_pan_dsc, self.current_epoch)
        self.logger.log('lesion_dsc', les_dsc, self.current_epoch)
        self.logger.log('micro_dice_score', micro_dscore, self.current_epoch)

    def do_split(self):
        """
        Custom split method that separates files based on their prefixes.
        Files starting with 'train' go to training set, files starting with 'validation' go to validation set.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # Get all case identifiers
        case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
        
        # Split based on prefix
        tr_keys = [case_id for case_id in case_identifiers if case_id.startswith('train')][:60]
        val_keys = [case_id for case_id in case_identifiers if case_id.startswith('validation')]
        
        # Log the split information
        self.print_to_log_file(f"Custom split based on prefixes:")
        self.print_to_log_file(f"Training cases: {len(tr_keys)}")
        self.print_to_log_file(f"Validation cases: {len(val_keys)}")
        
        if len(tr_keys) == 0 or len(val_keys) == 0:
            self.print_to_log_file("WARNING: No training or validation cases found. Please check your file naming convention.")
            
        return tr_keys, val_keys

    def visualize_validation_cases(self, num_cases=4):
        """
        Visualize segmentation results for a few validation cases.
        Args:
            num_cases: Number of validation cases to visualize
        """
        
        self.network.eval()
        with torch.no_grad():
            # Get a few validation cases
            val_cases = []
            for i, batch in enumerate(self.dataloader_val):
                if i >= 5:
                    break
                val_cases.append(batch)
            
            for case_idx, batch in enumerate(val_cases):
                data = batch['data'].to(self.device)
                target = batch['target']
                if isinstance(target, list):
                    target = target[0]  # Take first element if it's a list
                target = target.to(self.device)
                
                # Get predictions
                combined_output = self.network(data)
                seg_output, cls_output = self.network.unwrap_network_outputs(combined_output)
                if self.enable_deep_supervision:
                    seg_output = seg_output[0]
                
                # Convert to numpy arrays
                data = data.cpu().numpy()
                target = target.cpu().numpy()
                seg_pred = seg_output.argmax(1).cpu().numpy()
                
                # Create figure with GridSpec
                fig = plt.figure(figsize=(15, 5))
                gs = GridSpec(1, 3, figure=fig)
                
                # For 3D volumes, take middle slice
                slice_idx = data.shape[2] // 2
                
                # Plot input
                ax = fig.add_subplot(gs[0])
                ax.imshow(data[0, 0, slice_idx], cmap='gray')
                ax.set_title('Input')
                ax.axis('off')
                
                # Plot ground truth
                ax = fig.add_subplot(gs[1])
                ax.imshow(target[0, 0, slice_idx], cmap='viridis')
                ax.set_title('Ground Truth')
                ax.axis('off')
                
                # Plot prediction
                ax = fig.add_subplot(gs[2])
                ax.imshow(seg_pred[0, slice_idx], cmap='viridis')
                ax.set_title('Prediction')
                ax.axis('off')
                
                plt.suptitle(f'Validation Case {case_idx + 1}, Epoch {self.current_epoch}')
                
                # Save the figure
                save_path = os.path.join(self.output_folder, 'validation_visualizations')
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'epoch_{self.current_epoch}_case_{case_idx}.png'))
                plt.close()
        
    def run_training(self):
        '''
        Copy of run_training method from nnUNetTrainer.py with prints added
        '''
        self.on_train_start()

        self.print_to_log_file(f"""
            #######################################################################
            Hyperparameters used for training:
            Early stopping patience: {self.patience}
            Early stopping min delta: {self.min_delta}
            Classification loss weight: {self.cls_loss_weight} 
            #######################################################################
            """,
            also_print_to_console=True, add_timestamp=False)

        best_seg_loss = float('inf')
        best_cls_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []

            # Use tqdm to show epoch, batch and losses per iteration
            pbar = tqdm.tqdm(range(self.num_iterations_per_epoch), desc=f'Epoch {epoch} Training')
            for batch_id in pbar:
                train_outputs.append(self.train_step(next(self.dataloader_train)))
                collated_outputs = collate_outputs(train_outputs)

                # Update progress bar with current losses and accuracy
                pbar.set_postfix({
                    'Total Loss': f"{collated_outputs['loss'].mean():.4f}",
                    'Seg Loss': f"{collated_outputs['seg_loss'].mean():.4f}", 
                    'Cls Loss': f"{collated_outputs['cls_loss'].mean():.4f}",
                    'Cls Acc': f"{collated_outputs['cls_acc'].mean():.4f}"
                })

                # Print newline every self.num_iterations_per_epoch // 2 batches
                # if batch_id % (self.num_iterations_per_epoch // 2) == 0:
                #     print('\n')
            
            # Calculate average loss for this epoch
            current_seg_loss = np.mean([output['seg_loss'] for output in train_outputs])
            current_cls_loss = np.mean([output['cls_loss'] for output in train_outputs])

            # Early stopping logic
            if current_seg_loss < best_seg_loss - self.min_delta or current_cls_loss < best_cls_loss - self.min_delta:
                # Improvement
                best_seg_loss = current_seg_loss
                best_cls_loss = current_cls_loss
                patience_counter = 0
            else:  # No improvement
                patience_counter += 1
                
                if patience_counter >= self.patience:
                    self.print_to_log_file(f"Early stopping triggered at epoch {epoch}. No improvement for {self.patience} epochs.")
                    break
            
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
                
                self.visualize_validation_cases()

            self.on_epoch_end()

        self.on_train_end()

    def on_epoch_end(self):
        self.print_to_log_file(f'val_mean_fg_dice: {self.logger.my_fantastic_logging["mean_fg_dice"][-1]}')
        self.print_to_log_file(f'val_dice_per_class_or_region: {self.logger.my_fantastic_logging["dice_per_class_or_region"][-1]}')
        self.print_to_log_file(f'val_losses: {self.logger.my_fantastic_logging["val_losses"][-1]}')
        self.print_to_log_file(f'val_class_accuracy: {self.logger.my_fantastic_logging["class_accuracy"][-1]}')
        self.print_to_log_file(f'val_macro_f1: {self.logger.my_fantastic_logging["macro_f1"][-1]}')
        self.print_to_log_file(f'val_whole_pancreas_dsc: {self.logger.my_fantastic_logging["whole_pancreas_dsc"][-1]}')
        self.print_to_log_file(f'val_lesion_dsc: {self.logger.my_fantastic_logging["lesion_dsc"][-1]}')
        self.print_to_log_file(f'val_micro_dice_score: {self.logger.my_fantastic_logging["micro_dice_score"][-1]}')

        super().on_epoch_end()

    def perform_actual_validation(self, save_probabilities: bool = False, disable_tta: bool = False):
        start_time = time.time()
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictorWithClassification(tile_step_size=0.5, use_gaussian=True, use_mirroring=disable_tta,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False, num_classification_classes=self.num_classes)
        
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []; outputs_for_metrics = []; case_times = []
            
            # Show progress bar and time prediction
            pbar = tqdm.tqdm(enumerate(dataset_val.identifiers), total=len(dataset_val.identifiers),
                             desc="Generating predictions")
            for i, k in pbar:
                
                case_start_time = time.time()
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, targets , seg_prev, properties = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                # Time prediction, update progress bar
                start_time = time.time()
                prediction = predictor.predict_sliding_window_return_logits(data)
                end_time = time.time()
                # Update progress bar
                pbar.set_postfix({
                    'Inference time': f"{end_time - start_time} seconds"
                })
                
                prediction = prediction.cpu()
                # Add batch dimension to prediction
                prediction = prediction.unsqueeze(0)
                prediction, cls_output = self.network.unwrap_network_outputs(prediction, mean_cls=True)
                
                # seg_output_seg = prediction.argmax(1)[:, None].numpy()
                
                cls_label = int(k.split('_')[2])
                cls_pred = cls_output.argmax(1).cpu().numpy()                
                data = data.to(self.device, non_blocking=True)

                # Only for classification metric. Segmentation metrics are saved and read from files
                outputs_for_metrics_per_run = {
                    'class_pred' : cls_pred,
                    'class_labels' : cls_label
                }
                
                outputs_for_metrics.append(outputs_for_metrics_per_run)
                # this needs to go into background processes
                prediction = prediction.squeeze(0)

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                results.append(
                    segmentation_export_pool.starmap_async(
                        export_cls_prediction_from_logits, (
                            (cls_pred, k, join(validation_output_folder, 'subtype_results.csv')),
                        )
                    )
                )

                case_time = time.time() - case_start_time
                case_times.append(case_time)
                self.print_to_log_file(f"Time taken for case {k}: {case_time:.2f} seconds")

                # for debug purposes
                # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                #      self.dataset_json, output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        # next stage may have a different dataset class, do not use self.dataset_class
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

            
        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            ## Adding our metrics here 

            # collated_outputs = {}
            # for k in outputs_for_metrics[0].keys():
            #     collated_outputs[k] = [o[k] for o in outputs_for_metrics]
            
            collated_outputs = collate_outputs(outputs_for_metrics)
            class_pred = collated_outputs['class_pred']
            class_labels = collated_outputs['class_labels']

            # Macro F1
            macro_f1 = macro_f1_score(class_pred, class_labels, [0,1,2])
        
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)

            # dice = 2*tp / (2*tp + fp + fn)

            lesion_cls_idx = self.dataset_json['labels']['lesion pancreas']
            les_tp, les_fp, les_fn, les_tn = metrics['mean'][lesion_cls_idx]['TP'], metrics['mean'][lesion_cls_idx]['FP'],\
                metrics['mean'][lesion_cls_idx]['FN'], metrics['mean'][lesion_cls_idx]['TN']
            
            les_dsc = 2*les_tp / (2*les_tp + les_fp + les_fn)

            whole_pan_tp, whole_pan_fp, whole_pan_fn, whole_pan_tn = metrics['foreground_mean']['TP'], metrics['foreground_mean']['FP'],\
                metrics['foreground_mean']['FN'], metrics['foreground_mean']['TN']
            
            whole_pan_dsc = 2*whole_pan_tp / (2*whole_pan_tp + whole_pan_fp + whole_pan_fn)

            total_time = time.time() - start_time
            avg_time_per_case = sum(case_times) / len(case_times) if case_times else 0

            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file(f"Classification Macro F1: {macro_f1}")
            self.print_to_log_file(f"Whole Pancreas DSC: {whole_pan_dsc}")
            self.print_to_log_file(f"Lesion DSC: {les_dsc}")
            self.print_to_log_file(f"Total validation time: {total_time:.2f} seconds")
            self.print_to_log_file(f"Average time per case: {avg_time_per_case:.2f} seconds")

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
        
        print('Final Validation complete!!')
