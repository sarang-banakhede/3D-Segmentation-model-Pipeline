import os
import yaml
import torch
import json
import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import zoom
from tqdm import tqdm
from Utils.Swin_unetr import SwinUNETR
from Utils.Evaluation import evaluate

def ensure_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_files(folder, extensions):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(extensions)])

def normalize_volume(volume):
    return (volume - volume.min()) / (volume.max() - volume.min()) if volume.max() - volume.min() > 1e-8 else np.zeros_like(volume, dtype=np.float32)

def binarize_annotation(annotation, threshold=0.5):
    return (annotation > threshold).astype(np.uint8)

def load_volume(file_path):
    if file_path.lower().endswith(('.nii', '.nii.gz')):
        return np.transpose(nib.load(file_path).get_fdata(), (2, 0, 1))
    elif file_path.lower().endswith('.dcm'):
        dicom = pydicom.dcmread(file_path)
        return pydicom.pixel_data_handlers.util.apply_voi_lut(dicom.pixel_array, dicom)
    raise ValueError(f"Unsupported file format: {file_path}")

def resize_volume(volume, target_size):
    zoom_factors = (1, target_size[0] / volume.shape[1], target_size[1] / volume.shape[2])
    return zoom(volume, zoom_factors, order=1)

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
    return [volume[i:i + chunk_slices] for i in range(0, volume.shape[0], chunk_slices)]

def save_nifti(volume, save_path):
    nib.save(nib.Nifti1Image(np.transpose(volume, (1, 2, 0)).astype(np.float32), np.eye(4)), save_path)

def run_inference():
    config = yaml.safe_load(open("config.yaml", "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_images_folder = os.path.join(config["Paths"]["preprocessed_data_path"], 'Test', 'images')
    test_labels_folder = os.path.join(config["Paths"]["preprocessed_data_path"], 'Test', 'annotations')
    output_folder = os.path.join(config["Paths"]["result_path"], 'Inference')
    ensure_directories([output_folder])
    model_params = config['Model']
    model = SwinUNETR(
                    img_size=model_params["img_size"],
                    in_channels=model_params["in_channels"],
                    out_channels=model_params["out_channels"],
                    feature_size=model_params["feature_size"],
                    depths=model_params["depths"],
                    num_heads=model_params["num_heads"],
                    norm_name=model_params["norm_name"],
                    drop_rate=model_params["drop_rate"],
                    attn_drop_rate=model_params["attn_drop_rate"],
                    dropout_path_rate=model_params["dropout_path_rate"],
                    normalize=model_params["normalize"],
                    spatial_dims=model_params["spatial_dims"],
                    downsample=model_params["downsample"],
                    use_v2=model_params["use_v2"]).to(device)
    state_dict = torch.load(os.path.join(config['Paths']['result_path'], config["model_save_name"]), map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()} if all(k.startswith('module.') for k in state_dict.keys()) else state_dict)
    model.eval()
    
    target_size, chunk_slices, threshold = config["target_size"], config["Slice_in_one_chunk"], config["Training"].get("threshold", 0.5)
    
    for img_file in tqdm(get_files(test_images_folder, ('.nii', '.nii.gz')), desc="Processing test images"):
        base_name = os.path.splitext(os.path.splitext(img_file)[0])[0]
        img_output_dir = os.path.join(output_folder, base_name)
        ensure_directories([img_output_dir])
        
        image_path = os.path.join(test_images_folder, img_file)
        label_path = os.path.join(test_labels_folder, f"{base_name}.nii.gz")
           
        if not os.path.exists(label_path):
            label_path = os.path.join(test_labels_folder, f"{base_name}.nii")
            if not os.path.exists(label_path):
                print(f"Annotation not found for {img_file}, skipping.")
                continue
        
        volume, label = map(load_volume, (image_path, label_path))
        original_size = (volume.shape[1], volume.shape[2])
        volume_resized, label_resized = map(lambda x: resize_volume(normalize_volume(x), target_size), (volume, label))
        volume_uint8 = (volume_resized * 255).astype(np.uint8)
        
        volume_padded = pad_to_chunk(volume_uint8, chunk_slices)
        chunks = split_into_chunks(volume_padded, chunk_slices)
        
        pred_chunks = []
        for chunk in chunks:
            tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(1).to(device)
            with torch.no_grad():
                pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
            pred_chunks.append(pred)
        
        pred_volume = np.concatenate(pred_chunks, axis=0)[:volume_resized.shape[0]]
        
        pred_volume_resized = resize_volume(pred_volume, original_size)
        label_resized = resize_volume(label_resized, original_size)
        
        pred_binary, gt_binary = map(lambda x: binarize_annotation(x, threshold), (pred_volume_resized, label_resized))
        metrics = evaluate(pred_binary, gt_binary, threshold=threshold)
        
        for name, data in zip(['input', 'gt', 'pred', 'pred_raw'], [volume, gt_binary, pred_binary, pred_volume_resized]):
            save_nifti(data, os.path.join(img_output_dir, f"{base_name}_{name}.nii.gz"))
        
        with open(os.path.join(img_output_dir, f"{base_name}_metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
    print("Inference completed successfully.")

if __name__ == "__main__":
    run_inference()