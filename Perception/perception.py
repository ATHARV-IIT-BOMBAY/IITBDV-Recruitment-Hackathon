import cv2
import os
from ultralytics import YOLO

# Load the specific model you just downloaded
model = YOLO('YOLOv11s-Carmaker.pt')

# Load the test image from your folder
image_path = 'test_image.jpg' # Ensure this matches the downloaded image name

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found!")
else:
    img = cv2.imread(image_path)
    results = model(img)

    # Physical constants provided in the assignment
    H_m = 0.30  # Real-world height: 30 cm [cite: 468]
    f = 1000    # Focal length: 1000 mm [cite: 469]

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        h_pixels = y2 - y1 # Height in pixels derived from YOLO box 
        
        # Depth formula: d = (H * f) / h 
        distance_m = (H_m * f) / h_pixels
        
        # Draw bounding box and distance label 
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"Dist: {distance_m:.2f}m", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the final result
    cv2.imwrite('annotated_cones.jpg', img)
    print("Success! Result saved to annotated_cones.jpg")