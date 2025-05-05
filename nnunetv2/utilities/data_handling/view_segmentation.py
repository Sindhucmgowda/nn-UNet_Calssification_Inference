import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse

def view_segmentation(image_path: str, label_path: str = None, slice_idx: int = None):
    """
    View a medical image and its segmentation map.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the segmentation file (optional)
        slice_idx: Index of the slice to view (optional)
    """
    # Read the image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    
    # Read the segmentation map if provided
    seg_array = None
    if label_path and os.path.exists(label_path):
        seg = sitk.ReadImage(label_path)
        seg_array = sitk.GetArrayFromImage(seg)
    
    # If no slice index is provided, use the middle slice
    if slice_idx is None:
        slice_idx = image_array.shape[0] // 2
    
    # Create figure
    fig, ax = plt.subplots(1, 2 if seg_array is not None else 1, figsize=(15, 5))
    
    # Plot image
    if seg_array is not None:
        # Plot original image
        ax[0].imshow(image_array[slice_idx], cmap='gray')
        ax[0].set_title('Original Image')
        
        # Plot segmentation map with different colors for each label
        # Create a colored overlay
        colored_overlay = np.zeros((*seg_array[slice_idx].shape, 3))
        
        # Define colors for each label
        colors = {
            0: [0, 0, 0],        # Black for background
            1: [1, 0, 0],        # Red for label 1
            2: [0, 1, 0],        # Green for label 2
        }
        
        # Apply colors to each label
        for label_id, color in colors.items():
            mask = seg_array[slice_idx] == label_id
            for c in range(3):
                colored_overlay[..., c][mask] = color[c]
        
        # Overlay segmentation on original image
        ax[1].imshow(image_array[slice_idx], cmap='gray')
        ax[1].imshow(colored_overlay, alpha=0.5)  # alpha controls transparency
        ax[1].set_title('Segmentation Overlay')
        
        # Add legend with label statistics
        legend_elements = []
        for label_id, color in colors.items():
            if label_id in np.unique(seg_array):
                # Calculate percentage of this label in the current slice
                percentage = np.sum(seg_array[slice_idx] == label_id) / seg_array[slice_idx].size * 100
                legend_elements.append(
                    Patch(facecolor=color, 
                          label=f'Label {label_id}: {percentage:.1f}%')
                )
        ax[1].legend(handles=legend_elements, loc='upper right')
        
        # Print volume statistics
        print("\nVolume Statistics:")
        print("-----------------")
        total_voxels = np.prod(seg_array.shape)
        for label_id in np.unique(seg_array):
            voxel_count = np.sum(seg_array == label_id)
            percentage = voxel_count / total_voxels * 100
            print(f"Label {label_id}: {voxel_count} voxels ({percentage:.1f}%)")
    else:
        ax.imshow(image_array[slice_idx], cmap='gray')
        ax.set_title('Original Image')
    
    plt.suptitle(f'Slice {slice_idx} of {image_array.shape[0]}')
    plt.tight_layout()
    plt.show()

def view_dataset(dataset_dir: str, case_id: int = None):
    """
    View images and their segmentations from a nnUNet dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
        case_id: Specific case ID to view (optional)
    """
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    labels_tr_dir = os.path.join(dataset_dir, "labelsTr")
    
    if case_id:
        # View specific case
        image_path = os.path.join(images_tr_dir, f"case_{case_id:03d}_0000.nii.gz")
        label_path = os.path.join(labels_tr_dir, f"case_{case_id:03d}.nii.gz")
        
        if not os.path.exists(image_path):
            print(f"Case {case_id} not found!")
            return
        view_segmentation(image_path, label_path)
    else:
        # View all cases
        for file in os.listdir(images_tr_dir):
            if file.endswith("_0000.nii.gz"):
                case_id = file.split("_0000")[0]
                image_path = os.path.join(images_tr_dir, file)
                label_path = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")
                print(f"\nViewing case: {case_id}")
                view_segmentation(image_path, label_path)
                input("Press Enter to continue to next case...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View medical images and their segmentations from nnUNet dataset")
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    parser.add_argument("--case", help="Specific case ID to view (optional)")
    
    args = parser.parse_args()
    
    view_dataset(args.dataset_dir, args.case) 