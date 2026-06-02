import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model   import UNet
from unet_dataset import KidneyStoneSegDataset

TRAIN_IMG_DIR = "dataset/split/train/Stone"
TRAIN_LBL_DIR = "annotations_auto/train"
MODEL_SAVE    = "models/unet_stone.pth"
BATCH_SIZE    = 8
LR            = 1e-4
NUM_EPOCHS    = 15
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce   = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs        = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss    = 1 - (2 * intersection + self.smooth) / \
                           (probs.sum() + targets.sum() + self.smooth)

        return bce_loss + dice_loss


def main():
    print(f"Device: {DEVICE}")

    full_ds  = KidneyStoneSegDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, augment=True)
    val_size = max(1, int(0.15 * len(full_ds)))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        dice_sum = 0.0
        smooth   = 1e-6
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()

                probs = torch.sigmoid(preds) > 0.5
                inter = (probs * masks.bool()).float().sum()
                dice_sum += (2 * inter + smooth) / \
                            (probs.float().sum() + masks.sum() + smooth)

        val_loss /= len(val_loader)
        dice_val  = dice_sum / len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {dice_val:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✔ Best model saved (val_loss={val_loss:.4f})")

    print(f"\nTraining complete! Best model: {MODEL_SAVE}")


if __name__ == "__main__":
    main()
