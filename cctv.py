import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"runs\detect\train\weights\best.pt")

# Read the image
img_path = r"images\val\CCTV-7_81_jpg.rf.ca4f3f6b3862f28d208e414c34a410f4.jpg"
img = cv2.imread(img_path)

# Run inference
results = model(img)[0]  # Get the first prediction object

# Load class names from the model
class_names = model.names

# Draw bounding boxes and labels
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    cls_id = int(box.cls[0])               # Class ID
    conf = float(box.conf[0])              # Confidence score
    label = f"{class_names[cls_id]} {conf:.2f}"

    # Draw rectangle and label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

# Display the image
cv2.imshow("Labeled Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
