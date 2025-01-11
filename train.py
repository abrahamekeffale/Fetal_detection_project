import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import ShuffleSplit
import copy
import time
import os
import matplotlib.pyplot as plt
from data.dataset import FetalDataset
from data.transforms import transform_train, transform_val
from models.segnet import SegNet
from utils.metrics import loss_func, metrics_batch, dice_loss

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        pred = torch.sigmoid(output)
        _, metric_b = dice_loss(pred, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if metric_b is not None:
            running_metric += metric_b
        if sanity_check:
            break
    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric

def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    path2model = params["path2model"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        start_time = time.time()

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            torch.save(model, path2model)
            print("Copied best model weights and saved the model!")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        epoch_time = time.time() - start_time
        print("train loss: %.6f, dice: %.2f" % (train_loss, 100 * train_metric))
        print("val loss: %.6f, dice: %.2f" % (val_loss, 100 * val_metric))
        print("Epoch Time: %.2f seconds" % epoch_time)
        print("-" * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

# Parameters
params_model = {
    "input_shape": (1, 128, 192),
    "initial_filters": 16,
    "num_outputs": 1,
}

path2train = r"C:\Users\HP\Desktop\training_set"
path2weights = "./models/weights.pt"
path2model = "./models/model.pth"

# Load datasets
fetal_ds1 = FetalDataset(path2train, transform=transform_train)
fetal_ds2 = FetalDataset(path2train, transform=transform_val)

# Split datasets
sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = range(len(fetal_ds1))
for train_index, val_index in sss.split(indices):
    train_ds = Subset(fetal_ds1, train_index)
    val_ds = Subset(fetal_ds2, val_index)

# Data loaders
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, pin_memory=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegNet(params_model).to(device)

# Optimizer and scheduler
opt = Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

# Training parameters
params_train = {
    "num_epochs": 100,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2weights,
    "path2model": path2model,
}

# Train the model
model, loss_hist, metric_hist = train_val(model, params_train)

# Plot training history
num_epochs = params_train["num_epochs"]
plt.title("Train-Val Loss")
plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

plt.title("Train-Val Accuracy")
plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()