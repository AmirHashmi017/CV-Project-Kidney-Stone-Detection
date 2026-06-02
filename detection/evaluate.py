import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset import KidneyStoneDetectionDataset, collate_fn
from train import get_model
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def calculate_map(predictions, targets, iou_threshold=0.5):

    all_preds = []
    

    target_dict = {}
    
    for img_idx, (preds, targs) in enumerate(zip(predictions, targets)):
        p_boxes = preds['boxes'].cpu().numpy()
        p_scores = preds['scores'].cpu().numpy()
        p_labels = preds['labels'].cpu().numpy()
        
        for i in range(len(p_boxes)):
            all_preds.append({
                'img_idx': img_idx,
                'score': p_scores[i],
                'box': p_boxes[i],
                'label': p_labels[i]
            })
            
        t_boxes = targs['boxes'].cpu().numpy()
        t_labels = targs['labels'].cpu().numpy()
        target_dict[img_idx] = {'boxes': t_boxes, 'labels': t_labels, 'matched': [False]*len(t_boxes)}
 
    all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    num_ground_truths = sum(len(t['boxes']) for t in target_dict.values())
    
    if num_ground_truths == 0:
        return 0.0
        
    for i, pred in enumerate(all_preds):
        img_idx = pred['img_idx']
        pred_box = pred['box']
        
        targs = target_dict[img_idx]
        best_iou = 0
        best_target_idx = -1
        
        for j, t_box in enumerate(targs['boxes']):
            iou = compute_iou(pred_box, t_box)
            if iou > best_iou:
                best_iou = iou
                best_target_idx = j
                
        if best_iou >= iou_threshold:
            if not targs['matched'][best_target_idx]:
                tp[i] = 1
                targs['matched'][best_target_idx] = True 
            else:
                fp[i] = 1 
        else:
            fp[i] = 1
            
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / num_ground_truths
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
        
    return ap

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Evaluation on device: {device}")
    
    val_img_dir = "d:/CV-Project-Kidney-Stone-Detection/dataset/split/test/Stone"
    val_lbl_dir = "d:/CV-Project-Kidney-Stone-Detection/annotations_auto/test"
    
    val_dataset = KidneyStoneDetectionDataset(val_img_dir, val_lbl_dir, transform=True)
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('d:/CV-Project-Kidney-Stone-Detection/models/faster_rcnn_stone.pth', map_location=device))
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    results_dir = "d:/CV-Project-Kidney-Stone-Detection/results/predictions"
    os.makedirs(results_dir, exist_ok=True)
    
    print("Running inference and saving test results...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            preds = model(images)
            
            
            for j, (img_tensor, pred, targ) in enumerate(zip(images, preds, targets)):
                idx = i * 4 + j
                all_predictions.append(pred)
                all_targets.append(targ)
                
                
                img = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8).copy()
                
                p_boxes = pred['boxes'].cpu().numpy()
                p_scores = pred['scores'].cpu().numpy()
                t_boxes = targ['boxes'].cpu().numpy()
                
                
                for tbox in t_boxes:
                    x1, y1, x2, y2 = map(int, tbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
               
                for p_box, score in zip(p_boxes, p_scores):
                    if score > 0.5:
                        x1, y1, x2, y2 = map(int, p_box)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue in BGR
                        cv2.putText(img, f"Conf: {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                cv2.imwrite(os.path.join(results_dir, f"test_img_{idx}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    mAP_50 = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
    print(f"\n======================================")
    print(f"Mean Average Precision (mAP@0.5): {mAP_50:.4f}")
    print(f"======================================\n")
    
    with open("d:/CV-Project-Kidney-Stone-Detection/results/detection_metrics.txt", "w") as f:
        f.write("Object Detection Results\n")
        f.write("========================\n")
        f.write(f"mAP@0.5: {mAP_50:.4f}\n")

if __name__ == '__main__':
    main()
