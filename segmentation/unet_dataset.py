"""
Dataset for U-Net Segmentation Training
- Loads Stone images from the split directory
- Generates binary GT masks from bounding-box annotations
  (bbox region is filled white = stone; everything else black = background)
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_SIZE = 256   # U-Net input size (must be divisible by 16)


class KidneyStoneSegDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, augment=False):
        self.img_dir  = img_dir
        self.lbl_dir  = lbl_dir
        self.augment  = augment
        self.images   = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

        self.img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # ── Build GT mask from bbox annotations ──────────────────────────────
        base     = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base + ".txt")
        gt_mask  = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(gt_mask, (x1, y1), (x2, y2), 255, -1)

        # ── Resize both to fixed size ─────────────────────────────────────────
        img     = cv2.resize(img,     (IMG_SIZE, IMG_SIZE))
        gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_NEAREST)

        # Optional horizontal flip augmentation
        if self.augment and np.random.rand() > 0.5:
            img     = cv2.flip(img,     1)
            gt_mask = cv2.flip(gt_mask, 1)

        img_tensor  = self.img_tf(img)
        mask_tensor = torch.from_numpy((gt_mask > 0).astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor
