import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import nibabel as nib
import yaml
from scipy.ndimage import zoom

def ensure_directories(directories):
    """Ensure that the specified directories exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_files(folder, extensions):
    """Get a list of files in a folder with specified extensions."""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(extensions)])

def normalize_volume(volume):
    """Normalize the volume to the range [0, 1]."""
    return (volume - volume.min()) / (volume.max() - volume.min())

def binarize_annotation(annotation, threshold=0.5):
    """Binarize the annotation using a threshold."""
    return (annotation > threshold).astype(np.uint8)

def load_volume(file_path):
    """Load a volume from a file (NIfTI or DICOM)."""
    if file_path.lower().endswith(('.nii', '.nii.gz')):
        volume = nib.load(file_path).get_fdata()
        # Transpose the volume to match the expected format (slices, width, height)
        # From (width, height, slices) to (slices, width, height)
        volume = np.transpose(volume, (2, 0, 1))
        return volume
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def resize_volume(volume, target_size):
    """Resize the volume to the target size."""
    zoom_factors = (
        1,
        target_size[0] / volume.shape[1],
        target_size[1] / volume.shape[2]
    )
    return zoom(volume, zoom_factors, order=1)  # order=1 for bilinear interpolation

def pad_to_chunk(volume, chunk_slices):
    num_slices = volume.shape[0]
    padded_volume = volume.copy()
    
    if num_slices >= chunk_slices:
        target_slices = int(np.ceil(num_slices / chunk_slices) * chunk_slices)
    else:
        target_slices = chunk_slices
    
    while padded_volume.shape[0] < target_slices:
        next_slice_idx = padded_volume.shape[0] % num_slices
        next_slice = volume[next_slice_idx]
        padded_volume = np.concatenate([padded_volume, next_slice[np.newaxis, ...]], axis=0)
    
    return padded_volume

def split_into_chunks(volume, chunk_slices):
    """Split the volume into chunks of size chunk_slices."""
    return [volume[i:i + chunk_slices] for i in range(0, volume.shape[0], chunk_slices)]

def save_chunks(image_chunks, annotation_chunks, image_dir, annotation_dir, base_name):
    """Save image and annotation chunks."""
    for i, (img_chunk, ann_chunk) in enumerate(zip(image_chunks, annotation_chunks)):
        chunk_name = f"{base_name}_chunk{i}"
        # Transpose back to original format (width, height, slices) before saving
        # img_chunk_transposed = np.transpose(img_chunk, (1, 2, 0))
        # ann_chunk_transposed = np.transpose(ann_chunk, (1, 2, 0))
        nib.save(nib.Nifti1Image(img_chunk, np.eye(4)), os.path.join(image_dir, f"{chunk_name}.nii.gz"))
        nib.save(nib.Nifti1Image(ann_chunk, np.eye(4)), os.path.join(annotation_dir, f"{chunk_name}.nii.gz"))

def process_image_and_annotation(image_path, annotation_path, output_dir, chunk_slices, target_size):
    """Process a single image and its annotation."""
    image = load_volume(image_path)
    annotation = load_volume(annotation_path)

    image = resize_volume(image, target_size)
    annotation = resize_volume(annotation, target_size)

    image = normalize_volume(image)
    annotation = binarize_annotation(normalize_volume(annotation))

    image = pad_to_chunk(image, chunk_slices)
    annotation = pad_to_chunk(annotation, chunk_slices)

    image_chunks = split_into_chunks(image, chunk_slices)
    annotation_chunks = split_into_chunks(annotation, chunk_slices)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_chunks(image_chunks, annotation_chunks, output_dir["images"], output_dir["annotations"], base_name)

def split_data(image_folder, annotation_folder, output_folder, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    image_files = get_files(image_folder, ('.nii', '.nii.gz'))
    annotation_files = get_files(annotation_folder, ('.nii', '.nii.gz'))

    if len(image_files) != len(annotation_files):
        raise ValueError("The number of images and annotations must be equal.")

    train_images, test_images, train_annotations, test_annotations = train_test_split(
        image_files, annotation_files, test_size=test_size, random_state=random_state
    )

    dirs = {
        "train_images": os.path.join(output_folder, "Train/images"),
        "train_annotations": os.path.join(output_folder, "Train/annotations"),
        "test_images": os.path.join(output_folder, "Test/images"),
        "test_annotations": os.path.join(output_folder, "Test/annotations"),
    }
    ensure_directories(dirs.values())

    for file_list, src_folder, dest_folder in zip(
        [train_images, train_annotations, test_images, test_annotations],
        [image_folder, annotation_folder, image_folder, annotation_folder],
        [dirs["train_images"], dirs["train_annotations"], dirs["test_images"], dirs["test_annotations"]]
    ):
        for file_name in file_list:
            shutil.copy(os.path.join(src_folder, file_name), os.path.join(dest_folder, file_name))

    print(f"Data successfully split into Train and Test folders at {output_folder}")

def process_data(output_folder, chunk_slices, target_size):
    """Process train and test data."""
    dirs = {
        "train_images": os.path.join(output_folder, "Train/images"),
        "train_annotations": os.path.join(output_folder, "Train/annotations"),
        "test_images": os.path.join(output_folder, "Test/images"),
        "test_annotations": os.path.join(output_folder, "Test/annotations"),
        "processed_train_images": os.path.join(output_folder, "Processed/Train/images"),
        "processed_train_annotations": os.path.join(output_folder, "Processed/Train/annotations"),
        "processed_test_images": os.path.join(output_folder, "Processed/Test/images"),
        "processed_test_annotations": os.path.join(output_folder, "Processed/Test/annotations"),
    }
    ensure_directories(dirs.values())

    for split in ["train", "test"]:
        image_dir = dirs[f"{split}_images"]
        annotation_dir = dirs[f"{split}_annotations"]
        processed_image_dir = dirs[f"processed_{split}_images"]
        processed_annotation_dir = dirs[f"processed_{split}_annotations"]

        image_files = get_files(image_dir, ('.nii', '.nii.gz'))
        annotation_files = get_files(annotation_dir, ('.nii', '.nii.gz'))

        for image_file, annotation_file in zip(image_files, annotation_files):
            image_path = os.path.join(image_dir, image_file)
            annotation_path = os.path.join(annotation_dir, annotation_file)
            process_image_and_annotation(
                image_path, annotation_path,
                {"images": processed_image_dir, "annotations": processed_annotation_dir},
                chunk_slices, target_size
            )

    print(f"Data processing completed and saved in {output_folder}")

if __name__ == "__main__":
    
    config = yaml.safe_load(open("config.yaml"))
    image_folder = config['Paths']["dataset_images_path"]
    annotation_folder = config["Paths"]["dataset_annotation_path"]
    output_folder = config["Paths"]["preprocessed_data_path"]
    chunk_slices = config["Slice_in_one_chunk"]
    target_size = config["target_size"]
    test_split = config["test_split"]
    
    split_data(image_folder, annotation_folder, output_folder, test_split)
    process_data(output_folder, chunk_slices, target_size)