import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class KidneyStoneDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # We only consider Stone class as annotated, but there are subfolders in dataset/split/train.
        # So images_dir like: dataset/split/train/Stone
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # Load label
        base_name = os.path.basename(img_path)
        label_path = os.path.join(self.labels_dir, base_name.replace('.jpg', '.txt'))
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0]) # Since background is 0, we'll keep stone as 1
                        cx, cy, cw, ch = map(float, parts[1:])
                        
                        # Convert YOLO to [xmin, ymin, xmax, ymax]
                        xmin = (cx - cw / 2) * w
                        ymin = (cy - ch / 2) * h
                        xmax = (cx + cw / 2) * w
                        ymax = (cy + ch / 2) * h
                        
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(w, xmax)
                        ymax = min(h, ymax)
                        
                        # sometimes cw or ch could result in invalid boxes
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(1) # Faster R-CNN expects 0 as background, 1 as first foreground class
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            area = torch.empty((0,), dtype=torch.float32)
        
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transform is not None:
            # Need transform that doesn't break target bounding boxes.
            # ToTensor() standardizes 0-1 but we might need torchvision.transforms.functional
            # Here we just apply ToTensor manually to image
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)
            
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))
