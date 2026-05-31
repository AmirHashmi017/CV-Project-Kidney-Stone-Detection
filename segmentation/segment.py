"""
Kidney Stone Segmentation
Week 4 - CV Project

Method: Otsu Thresholding + Morphological Operations
- Converts CT/X-ray images to grayscale
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
- Uses Otsu's binarization to auto-threshold bright stone regions
- Applies morphological operations to clean up the mask
- Saves binary masks and visual overlays
- Calculates segmentation metrics (Dice Score, IoU, Precision, Recall)
"""

import cv2
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
STONE_IMG_DIR   = "dataset/split/test/Stone"
OUTPUT_MASKS    = "results/segmentation/masks"
OUTPUT_OVERLAYS = "results/segmentation/overlays"
METRICS_FILE    = "results/segmentation_metrics.txt"

os.makedirs(OUTPUT_MASKS, exist_ok=True)
os.makedirs(OUTPUT_OVERLAYS, exist_ok=True)

# ─── Segment one image ────────────────────────────────────────────────────────
def segment_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: CLAHE – improves stone visibility
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Step 3: Otsu thresholding – picks best threshold automatically
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 5: Keep only the largest contiguous region (likely the stone)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = mask

    return img, final_mask


def create_overlay(img, mask):
    """Overlay the mask on the original image as a semi-transparent green highlight."""
    overlay = img.copy()
    green_layer = np.zeros_like(img)
    green_layer[mask == 255] = (0, 255, 0)
    return cv2.addWeighted(overlay, 0.7, green_layer, 0.3, 0)


def dice_score(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt   = gt_mask.flatten().astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * intersection / denom) if denom > 0 else 1.0


def iou_score(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt   = gt_mask.flatten().astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    return (intersection / union) if union > 0 else 1.0


def pixel_precision_recall(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt   = gt_mask.flatten().astype(bool)
    tp   = np.logical_and(pred, gt).sum()
    fp   = np.logical_and(pred, ~gt).sum()
    fn   = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    image_paths = sorted(glob.glob(os.path.join(STONE_IMG_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} Stone images to segment.\n")

    dice_scores, iou_scores, precisions, recalls = [], [], [], []

    for i, img_path in enumerate(image_paths):
        img, mask = segment_image(img_path)
        if img is None:
            continue

        base = Path(img_path).stem

        # Save binary mask
        cv2.imwrite(os.path.join(OUTPUT_MASKS, f"{base}_mask.png"), mask)

        # Save colour overlay
        overlay = create_overlay(img, mask)
        cv2.imwrite(os.path.join(OUTPUT_OVERLAYS, f"{base}_overlay.jpg"), overlay)

        # Approximate ground truth from annotation bbox → filled ellipse as GT mask
        # (using the auto-annotation bboxes we generated earlier as proxy GT)
        lbl_path = f"annotations_auto/test/{base}.txt"
        h, w = img.shape[:2]
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(gt_mask, (x1, y1), (x2, y2), 255, -1)

        d  = dice_score(mask, gt_mask)
        iu = iou_score(mask, gt_mask)
        pr, re = pixel_precision_recall(mask, gt_mask)

        dice_scores.append(d)
        iou_scores.append(iu)
        precisions.append(pr)
        recalls.append(re)

        if i % 30 == 0:
            print(f"[{i+1}/{len(image_paths)}] {base}  Dice={d:.3f}  IoU={iu:.3f}  Prec={pr:.3f}  Rec={re:.3f}")

    # ── Summary ──
    mean_dice = np.mean(dice_scores)
    mean_iou  = np.mean(iou_scores)
    mean_prec = np.mean(precisions)
    mean_rec  = np.mean(recalls)

    summary = (
        "\n=== Kidney Stone Segmentation Metrics ===\n"
        f"Images processed : {len(dice_scores)}\n"
        f"Mean Dice Score  : {mean_dice:.4f}\n"
        f"Mean IoU Score   : {mean_iou:.4f}\n"
        f"Mean Precision   : {mean_prec:.4f}\n"
        f"Mean Recall      : {mean_rec:.4f}\n"
        "=========================================\n"
    )
    print(summary)

    with open(METRICS_FILE, "w") as f:
        f.write(summary)

    # ── Save metrics bar chart ──
    labels  = ["Dice", "IoU", "Precision", "Recall"]
    values  = [mean_dice, mean_iou, mean_prec, mean_rec]
    colors  = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("Kidney Stone Segmentation – Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/segmentation_metrics_chart.png", dpi=150)
    print("Metrics chart saved to results/segmentation_metrics_chart.png")
    print(f"Masks   saved to {OUTPUT_MASKS}/")
    print(f"Overlays saved to {OUTPUT_OVERLAYS}/")


if __name__ == "__main__":
    main()
