import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut  

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
        assert os.path.exists(mask_dir), f"Mask directory {mask_dir} does not exist."

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.nii', '.nii.gz'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.nii', '.nii.gz'))])

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be equal."

    def _load_image(self, path):
        try:
            if path.lower().endswith(('.nii', '.nii.gz')):
                data = nib.load(path).get_fdata()
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            return data
        
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = self._load_image(image_path)
        mask = self._load_image(mask_path)
        
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def create_dataloaders(Datapath, train_batch_size, test_batch_size, num_workers=0):

    train_image_dir = os.path.join(Datapath, "Processed", "Train", "images")
    train_mask_dir = os.path.join(Datapath, "Processed", "Train", "annotations")
    test_image_dir = os.path.join(Datapath, "Processed", "Test", "images")
    test_mask_dir = os.path.join(Datapath, "Processed", "Test", "annotations")

    train_dataset = CustomDataset(train_image_dir, train_mask_dir)
    test_dataset = CustomDataset(test_image_dir, test_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

if __name__ == '__main__':
    pass
    