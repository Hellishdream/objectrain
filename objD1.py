import cv2
import numpy as np
import torch

# Load your custom YOLO model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='/mouse_dataset/dataset.yaml')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/mouse_dataset/weights/best.pt')

def detect_mouse(image):
    # Run inference
    results = model(image)
    
    # Process results
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: x1, y1, x2, y2, confidence, class
    
    if len(detections) > 0:
        # Get the detection with highest confidence
        best_detection = detections[np.argmax(detections[:, 4])]
        x1, y1, x2, y2, confidence, class_id = best_detection
        
        # Calculate bounding box
        bounding_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
        # Get class name (assuming 'mouse' is the only class)
        class_name = "mouse"
        
        result_text = f"{class_name}: {confidence:.2f}"
        return result_text, bounding_box
    
    return None, None

def object_tracking():
    # Open the video camera
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()

        # Detect mouse in the frame
        result_text, bounding_box = detect_mouse(frame)

        if bounding_box is not None:
            frame = draw_bounding_box(result_text, bounding_box, frame)

        cv2.imshow("Mouse Detection and Tracking", frame)

        key = cv2.waitKey(1) & 0xff
        if key == 27:  # Exit if 'Esc' key is pressed
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Function to draw a bounding box and label on the frame
def draw_bounding_box(text, bounding_box, frame):
    x, y, w, h = bounding_box
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(text, font, 0.5, 1)
    text_origin = np.array([x, y - label_size[0][1]])

    cv2.rectangle(frame, tuple(text_origin), tuple(text_origin + label_size[0]),
                  color=(0, 0, 255), thickness=-1)

    cv2.putText(frame, text, (x, y - 5), font, 0.5, (255, 255, 255), 1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

if __name__ == "__main__":
    object_tracking()