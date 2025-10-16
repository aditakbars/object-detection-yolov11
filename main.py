# ...existing code...
from ultralytics import YOLO
import cv2
import time
import os
import sys
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Webcam object detector (YOLO)")
    p.add_argument("--model", default="yolo11n.pt", help="path to model weights")
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size (px)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default="cpu", help="device (cpu or cuda)")
    return p.parse_args()

def open_camera(source):
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # use DirectShow on Windows
    except Exception:
        cap = cv2.VideoCapture(source)
    return cap

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Download or place weights in the project folder, or pass --model path")
        sys.exit(1)

    # load model
    model = YOLO(args.model)

    cap = open_camera(args.source)
    if not cap.isOpened():
        print(f"Cannot open source: {args.source}")
        sys.exit(1)

    prev_time = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # fast resize to target width to improve FPS (maintain aspect)
            h, w = frame.shape[:2]
            if max(w, h) > args.imgsz:
                scale = args.imgsz / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_in = cv2.resize(frame, (new_w, new_h))
            else:
                frame_in = frame

            # inference + tracking
            results = model.track(frame_in, persist=True, conf=args.conf, imgsz=args.imgsz)

            # plot results (returns image with boxes)
            frame_out = results[0].plot()

            # compute and overlay FPS
            curr_time = time.perf_counter()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame_out, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("YOLO - press q to quit", frame_out)

            key = cv2.waitKey(15) & 0xFF
            if key == ord('s'):
                output_dir = "captures"
                os.makedirs(output_dir, exist_ok=True)
                fname = os.path.join(output_dir, f"capture_{int(time.time())}.png")
                cv2.imwrite(fname, frame_out)
                print("Saved", fname)
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# ...existing code...