import cv2
import time
import argparse
from ultralytics import YOLO

def run_inference(model_path, video_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow(f"{model_path} Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    run_inference(args.model, args.video)
