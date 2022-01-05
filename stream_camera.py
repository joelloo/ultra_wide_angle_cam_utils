import cv2
import numpy as np
import subprocess as sp
import time
import shlex
import argparse
import datetime
import os
import multiprocessing
import collections
from multiprocessing.managers import BaseManager


class ConcurrentRingBuffer:
    def __init__(self, maxlen=100):
        self.lock = multiprocessing.Lock()
        self.deque = collections.deque(maxlen=maxlen)

    def put(self, item):
        self.lock.acquire()
        self.deque.append(item)
        self.lock.release()

    def put_nowait(self, item):
        ret = self.lock.acquire(block=False)
        if ret:
            self.deque.append(item)
            self.lock.release()
        return ret

    def get(self):
        self.lock.acquire()
        if len(self.deque) == 0:
            self.lock.release()
            return None
        else:
            item = self.deque.popleft()
            self.lock.release()
            return item

    def get_nowait(self, item):
        ret = self.lock.acquire(block=False)
        if ret:
            if len(self.deque) == 0:
                self.lock.release()
                return True, None
            else:
                item = self.deque.popleft()
                self.lock.release()
                return True, item
        return False, None

def centre_image_logging_worker(camera_params, databuf):
    try:
        print("Centre worker: initialising")
        params = camera_params["centre"]
        stream_id = params["stream_id"]
        vis = params["vis"]
        log = params["log"]
        distort_type = params["log_distorted"]

        stream_id = stream_id + cv2.CAP_V4L2 if params["v4l2"] else stream_id
        cap = cv2.VideoCapture(stream_id)
        if not cap.isOpened():
            print("Centre worker: error opening camera")
            return
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    ch, cw = frame.shape[:2]
                    break

        mtx = params["camera_matrix"]
        dist = params["distortion"]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (cw, ch), cv2.CV_16SC2)
        ts_buffer = collections.deque(maxlen=60)
        last_printed = time.time()

        while True:
            ret, frame = cap.read()
            ts = int(round(time.time() * 1000))
            ts_buffer.append(float(ts) * 0.001)
            if not ret:
                print("Centre worker: dropped frame")
                continue

            if distort_type == 0:
                dst = frame
            else:
                dst = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            if log:
                if distort_type == 0:
                    databuf.put(("centre_distorted", ts, dst))
                elif distort_type == 1:
                    databuf.put((databuf.put(("centre", ts, dst))))
                elif distort_type == 2:
                    databuf.put(("centre_distorted", ts, frame))
                    databuf.put(("centre", ts, dst))
            if vis:
                cv2.imshow("Centre", dst)
                key = cv2.waitKey(1)

            curr = time.time()
            if curr - last_printed > 5 and len(ts_buffer) == 60:
                last_printed = curr
                print("Centre worker: ", 60.0 / (ts_buffer[-1] - ts_buffer[0]), " FPS")
    except KeyboardInterrupt:
        print("Centre worker: cleaning up and exiting")


def side_image_logging_worker(camera_params, databuf):
    try:
        print("Side worker: initialising")
        left_params = camera_params["left"]
        right_params = camera_params["right"]
        stream_id = left_params["stream_id"]
        vis = left_params["vis"]
        log = left_params["log"]
        distort_type = left_params["log_distorted"]

        stream_id = stream_id + cv2.CAP_V4L2 if params["v4l2"] else stream_id
        cap = cv2.VideoCapture(stream_id)
        if not cap.isOpened():
            print("Side worker: error opening camera")
            return
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    ch, cw = frame.shape[:2]
                    break

        left_mtx = left_params["camera_matrix"]
        left_dist = left_params["distortion"]
        left_crop_pos = max(0, int(np.floor(cw/2)) - left_params["crop_offset"])
        left_w = left_crop_pos
        lmap1, lmap2 = cv2.fisheye.initUndistortRectifyMap(left_mtx, left_dist, np.eye(3), left_mtx, (left_w, ch), cv2.CV_16SC2)

        right_mtx = right_params["camera_matrix"]
        right_dist = right_params["distortion"]
        right_crop_pos = min(cw-1, int(np.floor(cw/2)) + right_params["crop_offset"])
        right_w = cw - right_crop_pos
        rmap1, rmap2 = cv2.fisheye.initUndistortRectifyMap(right_mtx, right_dist, np.eye(3), right_mtx, (right_w, ch), cv2.CV_16SC2)

        left_write_dir = "left/"
        right_write_dir = "right/"
        ts_buffer = collections.deque(maxlen=60)
        last_printed = time.time()

        while True:
            ret, frame = cap.read()
            ts = int(round(time.time() * 1000))
            ts_buffer.append(float(ts) * 0.001)
            if not ret:
                print("Side worker: dropped frame")
                continue
            left_frame = frame[:, :left_crop_pos, :]
            right_frame = frame[:, right_crop_pos:, :]

            if distort_type == 0:
                left_dst = left_frame
                right_dst = right_frame
            else:
                left_dst = cv2.remap(left_frame, lmap1, lmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                right_dst = cv2.remap(right_frame, rmap1, rmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            if log:
                if distort_type == 0:
                    databuf.put(("left_distorted", ts, left_dst))
                    databuf.put(("right_distorted", ts, right_dst))
                elif distort_type == 1:
                    databuf.put(("left", ts, left_dst))
                    databuf.put(("right", ts, right_dst))
                if distort_type == 2:
                    databuf.put(("left_distorted", ts, left_frame))
                    databuf.put(("right_distorted", ts, right_frame))
                    databuf.put(("left", ts, left_dst))
                    databuf.put(("right", ts, right_dst))

            if vis:
                cv2.imshow("Left", left_dst[:, ::-1, :])
                cv2.imshow("Right", right_dst[:, ::-1, :])
                cv2.waitKey(1)

            curr = time.time()
            if curr - last_printed > 5 and len(ts_buffer) == 60:
                last_printed = curr
                print("Side worker: ", 60.0 / (ts_buffer[-1] - ts_buffer[0]), " FPS")
    except KeyboardInterrupt:
        print("Side worker: cleaning up and exiting")


def image_logging_writer(databuf):
    try:
        print("Spawned writer process")

        while True:
            ret = databuf.get()
            if ret:
                write_dir, ts, im = ret
                if not write_dir[:6] == "centre":
                    im = im[:, ::-1, :]
                cv2.imwrite(write_dir + '/' + str(ts) + '.jpg', im)
    except KeyboardInterrupt:
        print("Writer process cleaning up and exiting")


def centre_video_logging_worker(camera_params):
    write_process = None
    # count = 0
    try:
        print("Centre worker: initialising")
        params = camera_params["centre"]
        stream_id = params["stream_id"]
        vis = params["vis"]
        log = params["log"]
        distort_type = params["log_distorted"]

        stream_id = stream_id + cv2.CAP_V4L2 if params["v4l2"] else stream_id
        cap = cv2.VideoCapture(stream_id)
        if not cap.isOpened():
            print("Centre worker: error opening camera")
            return
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    ch, cw = frame.shape[:2]
                    break

        mtx = params["camera_matrix"]
        dist = params["distortion"]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (cw, ch), cv2.CV_16SC2)
        out_file = "centre_distorted/centre.mp4" if distort_type == 0 else "centre/centre.mp4"

        if log:
            print("Initialising FFMPEG writing process...")
            ffmpeg_exe = "ffmpeg"
            if params["arm64_exe_path"] is not None:
                ffmpeg_exe = params["arm64_exe_path"] + "/ffmpeg"
            write_process = sp.Popen(shlex.split(
                f'{ffmpeg_exe} -y -s {cw}x{ch} -pixel_format bgr24 -f rawvideo -framerate 30 -i pipe: -filter:v "settb=1/1000,setpts=RTCTIME/1000-1600000000000" -vcodec libx265 -pix_fmt yuv420p -crf 24 {out_file}'), 
                stdin=sp.PIPE,
                stderr=sp.DEVNULL,
                stdout=sp.DEVNULL)

        print("Reading and logging centre camera data...")
        ts_buffer = collections.deque(maxlen=60)
        last_printed = time.time()
        while True:
            ret, frame = cap.read()
            ts_buffer.append(time.time())
            if not ret:
                print("Centre worker: dropped frame")
                continue
            # count += 1

            if distort_type == 0:
                dst = frame
            else:
                dst = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            if log:
                write_process.stdin.write(dst.tobytes())
            if vis:
                cv2.imshow("Centre", dst)
                key = cv2.waitKey(1)

            curr = time.time()
            if curr - last_printed > 5 and len(ts_buffer) == 60:
                last_printed = curr
                print("Centre worker: ", 60.0 / (ts_buffer[-1] - ts_buffer[0]), " FPS")
    except KeyboardInterrupt:
        print("Centre worker: cleaning up and exiting")
        if write_process is not None:
            write_process.stdin.close()
            write_process.wait()
            write_process.terminate()
        # print("Centre worker received: ", str(count))
        print("Centre worker: Exited")


def side_video_logging_worker(camera_params):
    left_write_process = None
    right_write_process = None
    try:
        print("Side worker: initialising")
        left_params = camera_params["left"]
        right_params = camera_params["right"]
        stream_id = left_params["stream_id"]
        vis = left_params["vis"]
        log = left_params["log"]
        distort_type = left_params["log_distorted"]

        stream_id = stream_id + cv2.CAP_V4L2 if params["v4l2"] else stream_id
        cap = cv2.VideoCapture(stream_id)
        if not cap.isOpened():
            print("Side worker: error opening camera")
            return
        else:
            while True:
                ret, frame = cap.read()
                if ret:
                    ch, cw = frame.shape[:2]
                    break

        left_mtx = left_params["camera_matrix"]
        left_dist = left_params["distortion"]
        left_crop_pos = max(0, int(np.floor(cw/2)) - left_params["crop_offset"])
        left_w = left_crop_pos
        lmap1, lmap2 = cv2.fisheye.initUndistortRectifyMap(left_mtx, left_dist, np.eye(3), left_mtx, (left_w, ch), cv2.CV_16SC2)

        right_mtx = right_params["camera_matrix"]
        right_dist = right_params["distortion"]
        right_crop_pos = min(cw-1, int(np.floor(cw/2)) + right_params["crop_offset"])
        right_w = cw - right_crop_pos
        rmap1, rmap2 = cv2.fisheye.initUndistortRectifyMap(right_mtx, right_dist, np.eye(3), right_mtx, (right_w, ch), cv2.CV_16SC2)

        left_out_file = "left_distorted/left.mp4" if distort_type == 0  else "left/left.mp4"
        right_out_file = "right_distorted/right.mp4" if distort_type == 0 else "right/right.mp4"

        print("Initialising FFMPEG writing process...")
        if log:
            ffmpeg_exe = "ffmpeg"
            if params["arm64_exe_path"] is not None:
                ffmpeg_exe = params["arm64_exe_path"] + "/ffmpeg"
            left_write_process = sp.Popen(shlex.split(
                f'{ffmpeg_exe} -y -s {left_w}x{ch} -pixel_format bgr24 -f rawvideo -framerate 30 -i pipe: -filter:v "settb=1/1000,setpts=RTCTIME/1000-1600000000000" -vcodec libx265 -pix_fmt yuv420p -crf 24 {left_out_file}'), 
                stdin=sp.PIPE,
                stderr=sp.DEVNULL,
                stdout=sp.DEVNULL)

            right_write_process = sp.Popen(shlex.split(
                f'{ffmpeg_exe} -y -s {right_w}x{ch} -pixel_format bgr24 -f rawvideo -framerate 30 -i pipe: -filter:v "settb=1/1000,setpts=RTCTIME/1000-1600000000000" -vcodec libx265 -pix_fmt yuv420p -crf 24 {right_out_file}'), 
                stdin=sp.PIPE,
                stderr=sp.DEVNULL,
                stdout=sp.DEVNULL)

        print("Reading and logging side camera data...")
        ts_buffer = collections.deque(maxlen=60)
        last_printed = time.time()
        while True:
            ret, frame = cap.read()
            ts_buffer.append(time.time())
            if not ret:
                print("Side worker: dropped frame")
                continue
            left_frame = frame[:, :left_crop_pos, :]
            right_frame = frame[:, right_crop_pos:, :]

            if distort_type == 0:
                left_dst = left_frame[:, ::-1, :]
                right_dst = right_frame[:, ::-1, :]
            else:
                left_dst = cv2.remap(left_frame[:, ::-1, :], lmap1, lmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                right_dst = cv2.remap(right_frame[:, ::-1, :], rmap1, rmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            if log:
                left_write_process.stdin.write(left_dst.tobytes())
                right_write_process.stdin.write(right_dst.tobytes())
            if vis:
                cv2.imshow("Left", left_dst)
                cv2.imshow("Right", right_dst)
                cv2.waitKey(1)

            curr = time.time()
            if curr - last_printed > 5 and len(ts_buffer) == 60:
                last_printed = curr
                print("Side worker: ", 60.0 / (ts_buffer[-1] - ts_buffer[0]), " FPS")

    except KeyboardInterrupt:
        print("Side worker: cleaning up and exiting")
        if left_write_process is not None:
            left_write_process.stdin.close()
            left_write_process.wait()
            left_write_process.terminate()

        if right_write_process is not None:
            right_write_process.stdin.close()
            right_write_process.wait()
            right_write_process.terminate()

        print("Side worker: Exited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_calib", type=str, help="left camera calibration",
        default="../left_calibration.npz")
    parser.add_argument("--centre_calib", type=str, help="centre camera calibration",
        default="../centre_calibration.npz")
    parser.add_argument("--right_calib", type=str, help="right camera calibration",
        default="../right_calibration.npz")
    parser.add_argument("--centre_id", type=int, help="provide the integer identifying the unique port ID of centre camera, default being 0 (/dev/video0)",
        default=0)
    parser.add_argument("--side_id", type=int, help="provide the integer identifying the unique port ID of side cameras, default being 0 (/dev/video1)",
        default=1)
    parser.add_argument("--side_camera_crop_offset", type=int, help="no. of pixels from vertical centreline of image that is ignored when cropping side image",
        default=50)
    parser.add_argument("--visualise", type=bool, help="visualise the camera streams if true",
        default=False)
    parser.add_argument("--log_video", type=bool, help="log camera stream as MP4 with HEVC encoding if true, sequence of JPG images if false",
        default=False)
    parser.add_argument("--centre_max_fps", type=float, help="maximum FPS of centre camera",
        default=30)
    parser.add_argument("--log_dir", type=str, help="specify directory to log to",
        default=None)
    parser.add_argument("--log_data", type=bool, help="logs data to log_dir if true",
        default=True)
    parser.add_argument("--jpg_writer_pool_size", type=int, help="size of thread pool for writing data to file (only used for jpg writing)",
        default=3)
    parser.add_argument("--force_v4l2", type=bool, help="forces OpenCV to use v4l2 drivers if true (suitable for Linux), otherwise allows OpenCV to decide",
        default=False)
    parser.add_argument("--target_arm64", type=bool, help="uses binaries targeted at arm64 if true: e.g. uses prepackaged arm64 ffmpeg, ffprobe binaries etc.",
        default=False)
    parser.add_argument("--log_distorted", type=int, help="0: logs only distorted images, 1: logs only undistorted images, 2 (APPLICABLE ONLY TO JPG LOGGING!): logs both distorted and undistorted images",
        default=1)

    args = parser.parse_args()

    # Get directory the script is in, so that we can access the packaged ffmpeg and ffprobe utilities
    arm64_exe_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ffmpeg-4.4.1-arm64-static") if args.target_arm64 else None
    camera_params = {"left":{"calib_path":args.left_calib, "stream_id":args.side_id, "crop_offset":args.side_camera_crop_offset, 
                             "vis":args.visualise, "log":args.log_data, "v4l2":args.force_v4l2, "arm64_exe_path":arm64_exe_path,
                             "log_distorted":args.log_distorted}, 
                     "right":{"calib_path":args.right_calib, "stream_id":args.side_id, "crop_offset":args.side_camera_crop_offset, 
                              "vis": args.visualise, "log":args.log_data, "v4l2":args.force_v4l2, "arm64_exe_path":arm64_exe_path,
                              "log_distorted":args.log_distorted}, 
                     "centre":{"calib_path":args.centre_calib, "stream_id":args.centre_id, "vis":args.visualise, "log":args.log_data, 
                               "v4l2":args.force_v4l2, "arm64_exe_path":arm64_exe_path, "log_distorted":args.log_distorted}}

    if arm64_exe_path is not None and not os.path.exists(arm64_exe_path):
        print("Unpacking packaged ffmpeg")
        sp.run(["tar", "-xvf", "ffmpeg-release-arm64-static.tar.xz"])

    # Default directory for logging uses present date and time
    if args.log_dir is None:
        curr_time = datetime.datetime.now()
        args.log_dir = curr_time.strftime("%Y_%m_%d__%H_%M_%S")
        os.mkdir(args.log_dir)
        print("Logging to directory: ", args.log_dir)

    # Create separate folders for each camera
    os.chdir(args.log_dir)
    for cam in camera_params.keys():
        if args.log_distorted == 0:
            os.mkdir(cam + "_distorted")
        elif args.log_distorted == 1:
            os.mkdir(cam)
        else:
            os.mkdir(cam + "_distorted")
            os.mkdir(cam)

    # Load in the calibrations
    for cam, params in camera_params.items():
        calib = np.load(params["calib_path"])
        params['camera_matrix'] = calib['camera_matrix']
        params['distortion'] = calib['distortion']
        camera_params[cam] = params 

    print("Streaming and logging...")
    try:
        if args.log_video:
            # Logging as HEVC-encoded MP4 video
            centre_process = multiprocessing.Process(target=centre_video_logging_worker, args=(camera_params,))
            side_process = multiprocessing.Process(target=side_video_logging_worker, args=(camera_params,))
            processes = [centre_process]
            for process in processes:
                process.start()
            
            for process in processes:
                process.join()
        else:
            # Logging as JPG images
            BaseManager.register('ConcurrentRingBuffer', ConcurrentRingBuffer)
            manager = BaseManager()
            manager.start()
            buffer = manager.ConcurrentRingBuffer(maxlen=30)
            centre_process = multiprocessing.Process(target=centre_image_logging_worker, args=(camera_params,buffer))
            side_process = multiprocessing.Process(target=side_image_logging_worker, args=(camera_params, buffer))
            processes = [centre_process, side_process]

            for process in processes:
                process.start()

            # Start the consumer pool that writes the side images to file
            if args.log_data:
                pool_size = args.jpg_writer_pool_size
                consumer_pool = [multiprocessing.Process(target=image_logging_writer, args=(buffer,)) for _ in range(pool_size)]
                for process in consumer_pool:
                    process.start()
            
            for process in processes:
                process.join()

            if args.log_data:
                for process in consumer_pool:
                    process.join()
    except KeyboardInterrupt:
        print("Received interrupt...")

    print("Shutting down...")