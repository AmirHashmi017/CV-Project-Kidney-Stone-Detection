import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import KidneyStoneDetectionDataset, collate_fn

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    train_img_dir = "d:/CV-Project-Kidney-Stone-Detection/dataset/split/train/Stone"
    train_lbl_dir = "d:/CV-Project-Kidney-Stone-Detection/annotations_auto/train"
    

    train_dataset = KidneyStoneDetectionDataset(train_img_dir, train_lbl_dir, transform=True)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    

    model = get_model(num_classes=2)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 5 
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {losses.item():.4f}")
                
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} Average Loss: {epoch_loss/len(train_loader):.4f}")
        
    os.makedirs('d:/CV-Project-Kidney-Stone-Detection/models', exist_ok=True)
    torch.save(model.state_dict(), 'd:/CV-Project-Kidney-Stone-Detection/models/faster_rcnn_stone.pth')
    print("Model saved globally.")

if __name__ == '__main__':
    main()
