import cv2
import collections
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", type=int, help="camera ID", default=0)
    parser.add_argument("--force_v4l2", type=bool, help="forces OpenCV to use v4l2 drivers if true (suitable for Linux), otherwise allows OpenCV to decide",
        default=False)

    args = parser.parse_args()

    stream_id = args.cam_id + cv2.CAP_V4L2 if args.force_v4l2 else args.cam_id
    cap = cv2.VideoCapture(stream_id)
    if not cap.isOpened():
        print("Unable to connect to stream!")
        exit(0)
    else:
        ts_buffer = collections.deque(maxlen=60)
        last = time.time()
        curr = last
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Dropped frame")
                continue
            curr = time.time()
            ts_buffer.append(curr)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            if len(ts_buffer) == 60 and curr - last > 2:
                print("FPS: ", 60.0 / (ts_buffer[-1] - ts_buffer[0]))
                last = curr