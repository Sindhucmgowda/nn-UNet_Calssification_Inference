# nnUNet v2

This repository contains an implementation of nnUNet v2, a self-adapting framework for medical image segmentation. This implementations adopts the nnUNet to build a multi-task learning network - to perform both segmentation and classification. 

## Environments and Requirements

Install nnU-Net and MetricsReloaded repositories. You should meet the requirements mentioned in these repositories, this method does not need any additional requirements. 

### System Requirements
- Python 3.8 or higher
- CUDA 11.0 or higher
- GPU with at least 8GB VRAM (recommended)
- 16GB RAM minimum (32GB recommended)

### Installation
```bash
git clone https://github.com/MIC-DKFZ/nnUNet
pip install nnunetv2

git clone https://github.com/csudre/MetricsReloaded.git
pip install -e .
```

## Dataset Preparation
First organise the dataset and then perform pre-processing using different experiment planners.  

### Data Organization
Run the utilis/data_handling/prep_data.py file - this file reads the data and segmentation lables and places the datasets in the following structure:
```
nnUNet_raw/
├── Dataset001_Example/
│   ├── imagesTr/
│   ├── imagesTs/
│   ├── labelsTr/
│   └── dataset.json
```
### Data pre-processing
The prep_data.py file also performs preprossing on the raw dataset using different experiment planner files. My implementation uses two experiment planners. 

a. Using the default ResEncM encoder planner: 
```bash
!PYTHONPATH=./:$PYTHONPATH nnUNetv2_plan_and_preprocess -d DATASET_NAME_OR_ID -np 4 -pl nnUNetPlannerResEncM --verify_dataset_integrity
```

b. Using the custom ResEncM experiment planner with torch based re-sampling: 
```bash
!PYTHONPATH=./:$PYTHONPATH nnUNetv2_plan_and_preprocess -d DATASET_NAME_OR_ID -np 4 -pl nnUNetPlannerResEncM_torchres --verify_dataset_integrity -gpu_memory_target 16 
```

## Training
Run training on pre-processed dataset using different parameters. 

```bash
!PYTHONPATH=./:$PYTHONPATH nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD -tr TRAINER -p PLANS -device DEVICE
```

My implementation uses:
1. DATASET_NAME_OR_ID - Dataset001_pancreas_lesion
2. CONFIGURATION - 3d_fullres
3. FOLD - 0 ( Uses custom splitting to use provided training dataset for training and validation dataset for validation, skipping CV. This option just helps better result storage)
4. TRAINER - Uses custom data trainers that build new networks based on the nnUNet base-network and add different classification architectures  
    a. nnUNetTrainerWithClassification - Implements a simple classification head that extracts the feature from the last stage of segmentation encoder and uses it as input to predict lesion subtype labels. 

    b. nnUNetTrainerClsFeatureScaling - Implements a variant that does feature scaling for classification. This network scales encoder feature from the last stage to focus on region of interest before sending it to the classification head. It only use features near the lesion (knowledge embedded ROI) using gaussian smoothed lesion segmentation from the target, during training and predicted lesion seg during inference to help the classification head focus on the right region of interest.  

    c. nnUNetTrainerAgClassEnc - Implements a variant that incorporates attention mechanisms for classification. This network shares encoder features between the segmentation encoder and classification encoder using cross task attention gate. Each attention gate at depth d receives the feature maps from the classification encoder, and the feature maps extracted by the segmentation encoder at corresponding depth. Classification encoder features are then scaled based on the attention weights and the deepest feature is then scaled to focus on region of interest before sending it to the classification head.  

5. PLANS -  Use the following default and custom plans generated during pre-processing. On close examination, both plans are exactly the same except for the sampling technique - that changes inference but doesn't change the training.  
    a. nnUNetResEncUNetMPlans - Uses the default experiment planner for ResEncNet M encoder 
    b. nnUNetResEncUNetMPlans_torchres - ses the custom experiment planner for ResEncNet M encoder which uses torch based re-sampling for faster inference. 
6. DEVICE - cuda 


## Evaluation

The methods performance is evaluated on validation set   

```bash
!PYTHONPATH=./:$PYTHONPATH nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD -tr TRAINER -p PLANS -device DEVICE --val
```
Also added --disable-tta: diable test time augmentation to evaluate performance when mirroring is diabled for inference.  

### Metrics
We use the following metrics for evaluation. 

1. Dice coefficient for 
    - whole pancreas segmentation 
    - lesion segmentation
2. Macro F1 for classification. 

We implement these metrics we use dice scores and beta-F1 scores as implemented in the MetricsReloaded repository.   

## Inference

### Basic Inference
```bash
!PYTHONPATH=./:$PYTHONPATH nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -f FOLD -c CONFIGURATION -tr CUSTOM_TRAINER -f FOLD -p PLANS --predictor CUSTOM_PREDICTOR
```
Use the same parameters used during training and 

1. CUSTOM_PREDICTOR - nnUNetPredictorWithClassification - Implements a custom predictor function that can handle classification too.  

## Results

### Segmentation and Classification Performance 
| Model | Pancreas DSC | Lesion DSC  | Macro - F1 |
|---------|-------------|------------|-----------|
| nnUNetTrainerClsFeatureSharing | 0.818 | 0.685 | 0.181 |
| nnUNetTrainerAgClassEnc | 0.858 | 0.737 | 0.196 |

### Inference 

| Planner | Average Inference Time | Pancreas DSC | Lesion DSC |
|---------|-------------|------------|-----------|
| ResEncM | 5.98 | 0.858 | 0.713  |
| ResEnctorchres | 5.41 | 0.856 | 0.755 |
| ResEnctorchres no TTA | 1.8 | 0.86 | 0.764 |
