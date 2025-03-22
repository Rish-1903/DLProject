import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import itertools
from glob import glob
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler
import rasterio
from osgeo import gdal

# Parameters
dataset = "spot6"  # Choose dataset: "UAV", "SPOT6", or "S2"
norm_param = np.load('/kaggle/input/magicbethynet/MagicBathyNet/agia_napa/norm_param_spot6_an.npy')  # Normalization parameters
norm_param_depth = -30.443  # Depth normalization parameter
WINDOW_SIZE = (30, 30)  # Window size for sliding window
STRIDE = 2  # Stride for sliding window
BATCH_SIZE = 1  # Batch size
MAIN_FOLDER = '/kaggle/input/magicbethynet/MagicBathyNet/agia_napa'  # Path to your data folder
DATA_FOLDER = MAIN_FOLDER + '/img/spot6/img_{}.tif'  # Path to RGB images
LABEL_FOLDER = MAIN_FOLDER + '/depth/spot6/depth_{}.tif'  # Path to depth images
ERODED_FOLDER = MAIN_FOLDER + '/depth/spot6/depth_{}.tif'  # Path to eroded depth images
train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
test_images = ['411', '387', '410', '398', '370', '369', '397']

# Define the Physics-Inspired Neural Network (PINN)
class PhysicsInspiredNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PhysicsInspiredNN, self).__init__()
        self.n_channels = in_channels
        self.n_outputs = out_channels

        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(256 * (WINDOW_SIZE[0] // 4) * (WINDOW_SIZE[1] // 4), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels * WINDOW_SIZE[0] * WINDOW_SIZE[1])
        )

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = x.view(x.size(0), self.n_outputs, WINDOW_SIZE[0], WINDOW_SIZE[1])  # Reshape to output size
        return x

# Define the physics-based loss function
class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()

    def forward(self, output, target, mask):
        # Data-driven loss (MSE)
        mse_loss = F.mse_loss(output * mask, target * mask, reduction='sum') / mask.sum()

        # Physics-based loss (e.g., Laplacian smoothness)
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(output.device)
        laplacian = F.conv2d(output, laplacian_kernel, padding=1)
        physics_loss = torch.mean(laplacian ** 2)

        # Combine losses
        total_loss = mse_loss + 0.1 * physics_loss  # Adjust weight for physics loss
        return total_loss

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w)
    x2 = x1 + w
    y1 = random.randint(0, H - h)
    y2 = y1 + h
    return x1, x2, y1, y2

# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER, cache=False, augmentation=True):
        super(Dataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache
        self.data_files = [data_files.format(id) for id in ids]
        self.label_files = [label_files.format(id) for id in ids]
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return 10000  # Default epoch size

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2, 0, 1)), dtype='float32')
            data = (data - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis])
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            label = 1 / norm_param_depth * np.asarray(io.imread(self.label_files[random_idx]), dtype='float32')
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        if self.augmentation:
            data_p, label_p = self.data_augmentation(data_p, label_p)

        return torch.from_numpy(data_p), torch.from_numpy(label_p)

    @staticmethod
    def data_augmentation(*arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                array = array[::-1, :] if len(array.shape) == 2 else array[:, ::-1, :]
            if will_mirror:
                array = array[:, ::-1] if len(array.shape) == 2 else array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

# Load the network on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PhysicsInspiredNN(3, 1).to(device)

# Load the datasets
train_set = Dataset(train_images, DATA_FOLDER, LABEL_FOLDER, cache=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

# Define the optimizer
base_lr = 0.0001
optimizer = optim.Adam(net.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

# Training function
def train(net, optimizer, epochs, scheduler=None, save_epoch=15):
    criterion = PhysicsLoss()
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()

            # Forward pass
            output = net(data.float())

            # Compute loss
            target_mask = (target != 0).float().to(device)
            loss = criterion(output, target, target_mask)
            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                print(f'Epoch {e}/{epochs}, Iter {iter_}, Loss: {loss.item()}')

            iter_ += 1

        if e % save_epoch == 0:
            torch.save(net.state_dict(), f'model_epoch{e}.pth')

    torch.save(net.state_dict(), 'model_final.pth')

# Train the network
train(net, optimizer, 10, scheduler)

# Testing function
def test(net, test_ids):
    net.eval()
    all_preds, all_gts = [], []

    for id_ in test_ids:
        img = np.asarray(io.imread(DATA_FOLDER.format(id_)).transpose((2, 0, 1)), dtype='float32')
        img = (img - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis])
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(img_tensor.float()).cpu().numpy().squeeze()

        gt = 1 / norm_param_depth * np.asarray(io.imread(LABEL_FOLDER.format(id_)), dtype='float32')
        all_preds.append(pred)
        all_gts.append(gt)

    return all_preds, all_gts

# Test the network
all_preds, all_gts = test(net, test_images)

# Save the results
for pred, id_ in zip(all_preds, test_images):
    pred_img = pred * norm_param_depth
    plt.imshow(pred_img, cmap='viridis')
    plt.show()
    io.imsave(f'prediction_{id_}.tif', pred_img)
    
def calculate_metrics(predictions, ground_truths):
    """
    Calculate RMSE, MAE, and Standard Deviation between predictions and ground truths.
    
    Args:
        predictions (list of numpy arrays): List of predicted depth maps.
        ground_truths (list of numpy arrays): List of ground truth depth maps.
    
    Returns:
        rmse (float): Root Mean Squared Error.
        mae (float): Mean Absolute Error.
        std_dev (float): Standard Deviation of errors.
    """
    # Flatten the predictions and ground truths
    preds_flat = np.concatenate([p.ravel() for p in predictions])
    gts_flat = np.concatenate([g.ravel() for g in ground_truths])

    # Mask out invalid values (e.g., zeros in ground truth)
    mask = gts_flat != 0  # Assuming 0 represents invalid or missing data
    preds_flat = preds_flat[mask]
    gts_flat = gts_flat[mask]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((preds_flat - gts_flat) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(preds_flat - gts_flat))

    # Calculate Standard Deviation of errors
    std_dev = np.std(preds_flat - gts_flat)

    # Print the metrics
    print(f"RMSE: {rmse * -norm_param_depth:.3f} m")
    print(f"MAE: {mae * -norm_param_depth:.3f} m")
    print(f"Standard Deviation: {std_dev * -norm_param_depth:.3f} m")

    return rmse, mae, std_dev

# Test the network
all_preds, all_gts = test(net, test_images)

# Calculate and print metrics
rmse, mae, std_dev = calculate_metrics(all_preds, all_gts)
