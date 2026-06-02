import os, sys
import glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model   import UNet
from unet_dataset import KidneyStoneSegDataset, IMG_SIZE

TEST_IMG_DIR  = "dataset/split/test/Stone"
TEST_LBL_DIR  = "annotations_auto/test"
MODEL_PATH    = "models/unet_stone.pth"
OUT_MASKS     = "results/segmentation/unet_masks"
OUT_OVERLAYS  = "results/segmentation/unet_overlays"
METRICS_FILE  = "results/unet_segmentation_metrics.txt"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_MASKS,    exist_ok=True)
os.makedirs(OUT_OVERLAYS, exist_ok=True)


def dice(pred, gt, smooth=1e-6):
    p = pred.flatten().astype(bool)
    g = gt.flatten().astype(bool)
    inter = np.logical_and(p, g).sum()
    return (2 * inter + smooth) / (p.sum() + g.sum() + smooth)

def iou(pred, gt, smooth=1e-6):
    p = pred.flatten().astype(bool)
    g = gt.flatten().astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return (inter + smooth) / (union + smooth)

def prec_rec(pred, gt, smooth=1e-6):
    p = pred.flatten().astype(bool)
    g = gt.flatten().astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


def main():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    test_ds     = KidneyStoneSegDataset(TEST_IMG_DIR, TEST_LBL_DIR, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    img_paths   = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))

    dice_s, iou_s, prec_s, rec_s = [], [], [], []

    print(f"\nRunning inference on {len(test_ds)} test images...")
    with torch.no_grad():
        for i, (img_tensor, gt_tensor) in enumerate(test_loader):
            img_tensor = img_tensor.to(DEVICE)
            logits     = model(img_tensor)
            pred_mask  = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
            gt_mask    = (gt_tensor.squeeze().numpy() > 0).astype(np.uint8) * 255

            base = Path(img_paths[i]).stem

            orig  = cv2.imread(img_paths[i])
            oh, ow = orig.shape[:2]
            pred_full = cv2.resize(pred_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(OUT_MASKS, f"{base}_unet_mask.png"), pred_full)
            overlay    = orig.copy()
            green      = np.zeros_like(orig)
            green[pred_full == 255] = (0, 255, 0)
            overlay    = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)
            cv2.imwrite(os.path.join(OUT_OVERLAYS, f"{base}_unet_overlay.jpg"), overlay)

            d  = dice(pred_mask, gt_mask)
            iu = iou (pred_mask, gt_mask)
            pr, re = prec_rec(pred_mask, gt_mask)

            dice_s.append(d);  iou_s.append(iu)
            prec_s.append(pr); rec_s.append(re)

            if i % 30 == 0:
                print(f"[{i+1}/{len(test_ds)}] {base} | Dice={d:.3f} IoU={iu:.3f} "
                      f"Prec={pr:.3f} Rec={re:.3f}")

    m_dice = np.mean(dice_s); m_iou = np.mean(iou_s)
    m_prec = np.mean(prec_s); m_rec = np.mean(rec_s)

    summary = (
        "\n====== U-Net Kidney Stone Segmentation Metrics ======\n"
        f"Images tested  : {len(dice_s)}\n"
        f"Mean Dice Score: {m_dice:.4f}\n"
        f"Mean IoU Score : {m_iou:.4f}\n"
        f"Mean Precision : {m_prec:.4f}\n"
        f"Mean Recall    : {m_rec:.4f}\n"
        "=====================================================\n"
    )
    print(summary)
    with open(METRICS_FILE, "w") as f:
        f.write(summary)

    labels = ["Dice", "IoU", "Precision", "Recall"]
    vals   = [m_dice, m_iou, m_prec, m_rec]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("U-Net Kidney Stone Segmentation – Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/unet_metrics_chart.png", dpi=150)

    print("Metrics chart → results/unet_metrics_chart.png")
    print(f"Masks    → {OUT_MASKS}/")
    print(f"Overlays → {OUT_OVERLAYS}/")


if __name__ == "__main__":
    main()
