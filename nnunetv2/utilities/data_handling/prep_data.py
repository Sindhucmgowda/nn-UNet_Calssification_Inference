import os
import SimpleITK as sitk
import numpy as np
import shutil
import argparse
from pathlib import Path
import matplotlib.pyplot as plt        
from matplotlib.patches import Patch
from view_segmentation import view_dataset
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

def convert_data(source_dir, dataset_dir, dataset_id = 1, dataset_name = "Assesment"):

    # For validation split, we save everything to the training folder
    fin_img_dir = {'train':"imagesTr", 'validation':"imagesTr", 'test': "imagesTs"}
    fin_lab_dir = {'train':"labelsTr", 'validation':"labelsTr"}

    dataset_dir = os.path.join(dataset_dir, f'Dataset{dataset_id:03d}_{dataset_name}')
    
    # Process training data
    for split_type in ["train", "validation", "test"]:
        print(f'processing {split_type} data')

        images_dir = os.path.join(dataset_dir, fin_img_dir[split_type])
        os.makedirs(images_dir, exist_ok=True)

        if split_type != 'test':
            labels_dir = os.path.join(dataset_dir, fin_lab_dir[split_type])
            os.makedirs(labels_dir, exist_ok=True)

        split_dir = os.path.join(source_dir,split_type)

        for subtype in os.listdir(split_dir):
            subtype_dir = os.path.join(split_dir, subtype)

            if split_type == 'test':
                # unknown labels in test
                subtype_dir = split_dir

            if not os.path.isdir(subtype_dir):
                continue

            # Process each case in the subtype directory
            for fname in os.listdir(subtype_dir):
                if fname.endswith("_0000.nii.gz"):  # This is an image file

                    # Copy and rename image file
                    src_image = os.path.join(subtype_dir, fname)
                    dst_image = os.path.join(images_dir, f"{split_type}_{fname}")
                    shutil.copy2(src_image, dst_image)

                    # Copy and rename corresponding label file
                    if split_type != 'test':
                        # no labels in test
                        label_file = fname.replace('_0000', '')
                        src_label = os.path.join(subtype_dir, label_file)
                        dst_label = os.path.join(labels_dir, f"{split_type}_{label_file}")
                        
                        label = sitk.ReadImage(src_label)
                        # Convert labels to int64
                        label_array = np.int64(sitk.GetArrayFromImage(label))
                        
                        # Create new image
                        new_label = sitk.GetImageFromArray(label_array)
                        new_label.CopyInformation(label)  # Copy metadata
                        
                        # Save the label file
                        sitk.WriteImage(new_label, dst_label)        
                    
    return dataset_dir

def get_args():
    parser = argparse.ArgumentParser(description="Convert data to nnUNet format")
    parser.add_argument("--input_dir", help="Path to the input dataset directory", required=True)
    parser.add_argument("--output_dir", help="Path to the output dataset directory", required=True)
    parser.add_argument("--dataset_id", help="Dataset ID", type = int, required=False)
    parser.add_argument("--dataset_name", help="Dataset Name", required=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    nnUNet_raw_dir = os.path.join(args.output_dir, 'nnUNet_raw')
    nnUNet_preprocessed_dir = os.path.join(args.output_dir, 'nnUNet_preprocessed')
    nnUNet_results_dir = os.path.join(args.output_dir, 'nnUNet_results')
    os.environ['nnUNet_raw'] = nnUNet_raw_dir
    os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed_dir
    os.environ['nnUNet_results'] = nnUNet_results_dir
    
    # Create directories
    for dir_name in [nnUNet_raw_dir, nnUNet_preprocessed_dir, nnUNet_results_dir]:
        os.makedirs(dir_name, exist_ok=True)

    assert args.dataset_id is not None, "Dataset ID is required"
    assert args.dataset_name is not None, "Dataset Name is required"

    # Convert data
    dataset_dir = convert_data(args.input_dir, nnUNet_raw_dir, args.dataset_id, args.dataset_name) 
    
    # Generate dataset.json
    if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
        num_training_cases = len(os.listdir(os.path.join(dataset_dir, 'imagesTr')))
    
        generate_dataset_json(
            output_folder = dataset_dir,
            channel_names = {'0': 'CT'},
            labels = {'background': 0, 'normal pancreas': 1, 'lesion pancreas': 2},
            num_training_cases = num_training_cases,
            file_ending = '.nii.gz',
            citation = None,
            regions_class_order = None,
            dataset_name = args.dataset_name
        )
    
    # running plan and preprocess 
    os.system(f'nnUNetv2_plan_and_preprocess -d {args.dataset_id} -np 4 -pl nnUNetPlannerResEncM --verify_dataset_integrity') 

    # running plan and preprocess with torch re-sampling 
    os.system(f'nnUNetv2_plan_and_preprocess -d {args.dataset_id} -np 4 -pl nnUNetPlannerResEncM_torchres --verify_dataset_integrity -gpu_memory_target 16 ') 
    