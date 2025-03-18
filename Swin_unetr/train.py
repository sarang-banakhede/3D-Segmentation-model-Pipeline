import os
import torch
from torch import nn, optim
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from monai.losses import DiceCELoss  
from Utils.Swin_unetr import SwinUNETR
from Utils.Dataloader import create_dataloaders
from Utils.Evaluation import evaluate
import numpy as np
import json
import yaml
from tqdm import tqdm

# Load configuration
config = yaml.safe_load(open("config.yaml"))

# Check for multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")

# Define the model
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
    use_v2=model_params["use_v2"],
)

# Wrap the model with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

# Move the model to the device
model = model.to(device)

# Load weights if available
weight_path = config.get('intermediate_model_weight_path', None)
last_epoch_number = 0
if weight_path and weight_path.lower() != "none" and os.path.exists(weight_path):
    # Load the state dictionary
    state_dict = torch.load(weight_path)
    
    # If the model is wrapped with DataParallel, add the `module.` prefix to the keys
    if isinstance(model, DataParallel):
        # Add `module.` prefix to keys if it's not already present
        new_state_dict = {f"module.{k}" if not k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        # Remove the `module.` prefix from the keys
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    last_epoch_number = int(weight_path.split('\\')[-1].split('.')[0])
    print(f"Loaded weights from {weight_path}. Resuming from epoch {last_epoch_number}.")
else:
    print("No weights found. Starting from scratch.")

# Define loss function (DiceCELoss from MONAI)
# For binary segmentation with sigmoidal outputs and binary labels
criterion = DiceCELoss(sigmoid=True, to_onehot_y=False)

# Define optimizer
training_params = config['Training']
optimizer = getattr(optim, training_params["optimizer"])(model.parameters(), lr=training_params["learning_rate"])

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=training_params.get("step_size", 30), gamma=training_params.get("gamma", 0.1))

# Create directories for saving results
paths = config['Paths']
os.makedirs(os.path.join(paths["result_path"], "weights"), exist_ok=True)

# Load data
train_loader, test_loader = create_dataloaders(
    paths["preprocessed_data_path"],
    train_batch_size=config['Dataloader']["train_batch_size"],
    test_batch_size=config['Dataloader']["test_batch_size"],
    num_workers=config['Dataloader']["num_workers"]
)

# Load training and testing history
train_history_path = os.path.join(paths["result_path"], config["train_history_name"])
test_history_path = os.path.join(paths["result_path"], config["test_history_name"])

train_history = {}
test_history = {}

if os.path.exists(train_history_path):
    with open(train_history_path, "r") as train_file:
        train_history = json.load(train_file)
if os.path.exists(test_history_path):
    with open(test_history_path, "r") as test_file:
        test_history = json.load(test_file)

# Training loop
for epoch in tqdm(range(last_epoch_number, training_params["epochs"]), desc="Epochs"):
    model.train()
    train_losses, train_metrics_list = [], []

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        for pred, label in zip(outputs.detach().cpu().numpy(), labels.cpu().numpy()):
            metrics = evaluate(pred, label, threshold=training_params["threshold"])
            if metrics:
                train_metrics_list.append(metrics)

    if train_metrics_list:
        train_history[epoch + 1] = {metric: np.mean([m[metric] for m in train_metrics_list]) for metric in train_metrics_list[0].keys()}
        train_history[epoch + 1]["average_loss"] = np.mean(train_losses)
    else:
        train_history[epoch + 1] = {"average_loss": np.mean(train_losses)}

    # Validation
    model.eval()
    test_losses, test_metrics_list = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            for pred, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                metrics = evaluate(pred, label, threshold=training_params["threshold"])
                if metrics:
                    test_metrics_list.append(metrics)

    if test_metrics_list:
        test_history[epoch + 1] = {metric: np.mean([m[metric] for m in test_metrics_list]) for metric in test_metrics_list[0].keys()}
        test_history[epoch + 1]["average_loss"] = np.mean(test_losses)
    else:
        test_history[epoch + 1] = {"average_loss": np.mean(test_losses)}

    # Step the learning rate scheduler
    scheduler.step()

    # Save model and history periodically
    if (epoch + 1) % config['save_epoch'] == 0:
        # Save the model without DataParallel wrapper
        if isinstance(model, DataParallel):
            torch.save(model.module.state_dict(), os.path.join(paths["result_path"], "weights", f"{epoch + 1}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(paths["result_path"], "weights", f"{epoch + 1}.pth"))
        
        with open(train_history_path, "w") as train_file:
            json.dump({int(k): {m: float(v) for m, v in metrics.items()} for k, metrics in train_history.items()}, train_file, indent=4)
        with open(test_history_path, "w") as test_file:
            json.dump({int(k): {m: float(v) for m, v in metrics.items()} for k, metrics in test_history.items()}, test_file, indent=4)

# Save final model and history
if isinstance(model, DataParallel):
    torch.save(model.module.state_dict(), os.path.join(paths["result_path"], config["model_save_name"]))
else:
    torch.save(model.state_dict(), os.path.join(paths["result_path"], config["model_save_name"]))

with open(train_history_path, "w") as train_file:
    json.dump({int(k): {m: float(v) for m, v in metrics.items()} for k, metrics in train_history.items()}, train_file, indent=4)
with open(test_history_path, "w") as test_file:
    json.dump({int(k): {m: float(v) for m, v in metrics.items()} for k, metrics in test_history.items()}, test_file, indent=4)

print("Training complete. Model and history files saved.")