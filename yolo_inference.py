import cv2
import time
import argparse
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def run_inference(model_path, video_path, output_path=None, display=False):
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = None
    if output_path:
        logger.info(f"Saving output to: {output_path}")
        out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.debug("End of video stream or failed to read frame.")
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

        if out:
            out.write(frame)

        if display:
            cv2.imshow("YOLO Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Display interrupted by user.")
                break

    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()

    logger.info("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--display", action="store_true", help="Show real-time output window")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    run_inference(args.model, args.video, args.output, args.display)
