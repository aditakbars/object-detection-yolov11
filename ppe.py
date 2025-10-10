import cv2
import numpy as np
from ultralytics import YOLO

class PPEApp:
    def __init__(self):
        # Initialize the YOLO model
        self.model = YOLO("bestn.pt")  # Replace with your model path
        self.confidence_threshold = 0.8

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to access the webcam.")
            exit()

        # Set the video window size to fullscreen
        cv2.namedWindow("PPE Detection", cv2.WINDOW_NORMAL)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return

        # Perform YOLO inference with dynamic confidence
        results = self.model(frame, conf=self.confidence_threshold, iou=0.3)

        # Initialize counters
        hardhat_count = 0
        vest_count = 0
        glasses_count = 0

        # Count detected items
        if len(results[0].boxes) > 0:
            detections = results[0].boxes.cls
            for det in detections:
                label = results[0].names[int(det)]
                if label == 'hardhat':
                    hardhat_count += 1
                elif label == 'vest':
                    vest_count += 1
                elif label == 'safety glasses':
                    glasses_count += 1

        # Annotate frame
        annotated_frame = results[0].plot()

        # Status messages
        hardhat_status = 'Yes' if hardhat_count > 0 else 'No'
        vest_status = 'Yes' if vest_count > 0 else 'No'
        glasses_status = 'Yes' if glasses_count > 0 else 'No'

        # Display status with appropriate colors (only Yes/No in color)
        font_scale = 0.7  # Smaller font size
        thickness = 2     # Slightly thinner text

        # Hardhat
        cv2.putText(annotated_frame, "Hardhat: ", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, hardhat_status, 
                    (150, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0) if hardhat_status == 'Yes' else (0, 0, 255), thickness)

        # Vest
        cv2.putText(annotated_frame, "Vest: ", 
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, vest_status, 
                    (150, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0) if vest_status == 'Yes' else (0, 0, 255), thickness)

        # Safety Glasses
        cv2.putText(annotated_frame, "Safety Glasses: ", 
                    (50, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, glasses_status, 
                    (200, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0) if glasses_status == 'Yes' else (0, 0, 255), thickness)

        # Display the annotated frame
        cv2.imshow("PPE Detection", annotated_frame)

    def run(self):
        while True:
            self.update_frame()

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close any open windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = PPEApp()
    app.run()