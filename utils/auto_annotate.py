import cv2
import os
import glob

def auto_annotate(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    count = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply binary thresholding to find brightest spots (potential stones)
        # Using a high threshold since kidney stones often appear bright in scans
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img.shape[:2]
        labels = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter tiny spots (noise)
            if bw > 5 and bh > 5 and bw < w*0.5 and bh < h*0.5:
                # YOLO format: class x_center_norm y_center_norm width_norm height_norm
                cx = (x + bw/2.0) / w
                cy = (y + bh/2.0) / h
                nw = bw / w
                nh = bh / h
                labels.append(f"1 {cx} {cy} {nw} {nh}") # class 1 for stone
                
        # If no contours found with 200, try adaptive or lower threshold
        if len(labels) == 0:
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw > 10 and bh > 10 and bw < w*0.5 and bh < h*0.5:
                    cx = (x + bw/2.0) / w
                    cy = (y + bh/2.0) / h
                    nw = bw / w
                    nh = bh / h
                    labels.append(f"1 {cx} {cy} {nw} {nh}")
                    break # just take the largest/first one if we had to lower threshold
                    
        # Write to txt file
        base_name = os.path.basename(img_path).replace('.jpg', '.txt')
        with open(os.path.join(label_dir, base_name), 'w') as f:
            f.write("\n".join(labels))
        count += 1
        
    print(f"Annotated {count} images in {image_dir} -> {label_dir}")

def main():
    auto_annotate("d:/CV-Project-Kidney-Stone-Detection/dataset/split/train/Stone", "d:/CV-Project-Kidney-Stone-Detection/annotations_auto/train")
    auto_annotate("d:/CV-Project-Kidney-Stone-Detection/dataset/split/test/Stone", "d:/CV-Project-Kidney-Stone-Detection/annotations_auto/test")

if __name__ == "__main__":
    main()
