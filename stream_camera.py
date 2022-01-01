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
        # print("~",len(self.deque))
        self.lock.release()

    def put_nowait(self, item):
        ret = self.lock.acquire(block=False)
        if ret:
            self.deque.append(item)
            self.lock.release()
        return ret

    def get(self):
        self.lock.acquire()
        # print("=",len(self.deque))
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

def centre_image_logging_worker(camera_params):
    try:
        print("Centre worker: initialising")
        params = camera_params["centre"]
        stream_id = params["stream_id"]
        vis = params["vis"]
        log = params["log"]

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
        write_dir = "centre/"
        ts_buffer = collections.deque(maxlen=60)
        last_printed = time.time()

        while True:
            ret, frame = cap.read()
            ts = int(round(time.time() * 1000))
            ts_buffer.append(float(ts) * 0.001)
            if not ret:
                print("Centre worker: dropped frame")
                continue
            dst = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if log:
                cv2.imwrite(write_dir + str(ts) + '.jpg', dst)
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
            left_dst = cv2.remap(left_frame, lmap1, lmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right_dst = cv2.remap(right_frame, rmap1, rmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # left_dst = left_dst[:, ::-1, :]
            # right_dst = right_dst[:, ::-1, :]
            # cv2.imwrite(left_write_dir + str(ts) + '.jpg', left_dst)
            # cv2.imwrite(right_write_dir + str(ts) + '.jpg', right_dst)

            if log:
                databuf.put((True, ts, left_dst))
                databuf.put((False, ts, right_dst))

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


def side_image_logging_writer(databuf):
    try:
        print("Spawned writer process")
        print(os.getcwd())
        left_write_dir = "left/"
        right_write_dir = "right/"

        while True:
            ret = databuf.get()
            if ret:
                # print("Received, writing")
                is_left, ts, im = ret
                write_dir = left_write_dir if is_left else right_write_dir
                im = im[:, ::-1, :]
                cv2.imwrite(write_dir + str(ts) + '.jpg', im)
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
        out_file = "centre/centre.mp4"

        if log:
            print("Initialising FFMPEG writing process...")
            write_process = sp.Popen(shlex.split(
                f'ffmpeg -y -s {cw}x{ch} -pixel_format bgr24 -f rawvideo -framerate 30 -i pipe: -filter:v "settb=1/1000,setpts=RTCTIME/1000-1600000000000" -vcodec libx265 -pix_fmt yuv420p -crf 24 {out_file}'), 
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

        left_out_file = "left/left.mp4"
        right_out_file = "right/right.mp4"

        print("Initialising FFMPEG writing process...")
        if log:
            left_write_process = sp.Popen(shlex.split(
                f'ffmpeg -y -s {left_w}x{ch} -pixel_format bgr24 -f rawvideo -framerate 30 -i pipe: -filter:v "settb=1/1000,setpts=RTCTIME/1000-1600000000000" -vcodec libx265 -pix_fmt yuv420p -crf 24 {left_out_file}'), 
                stdin=sp.PIPE,
                stderr=sp.DEVNULL,
                stdout=sp.DEVNULL)

            right_write_process = sp.Popen(shlex.split(
                f'ffmpeg -y -s {right_w}x{ch} -pixel_format bgr24 -f rawvideo -framerate 30 -i pipe: -filter:v "settb=1/1000,setpts=RTCTIME/1000-1600000000000" -vcodec libx265 -pix_fmt yuv420p -crf 24 {right_out_file}'), 
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
            left_frame = frame[:, :left_crop_pos:-1, :]
            right_frame = frame[:, right_crop_pos::-1, :]
            left_dst = cv2.remap(left_frame, lmap1, lmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right_dst = cv2.remap(right_frame, rmap1, rmap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
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

    args = parser.parse_args()
    camera_params = {"left":{"calib_path":args.left_calib, "stream_id":args.side_id, "crop_offset":args.side_camera_crop_offset, "vis":args.visualise, "log":args.log_data}, 
                     "right":{"calib_path":args.right_calib, "stream_id":args.side_id, "crop_offset":args.side_camera_crop_offset, "vis": args.visualise, "log":args.log_data}, 
                     "centre":{"calib_path":args.centre_calib, "stream_id":args.centre_id, "vis":args.visualise, "log":args.log_data}}

    # Default directory for logging uses present date and time
    if args.log_dir is None:
        curr_time = datetime.datetime.now()
        args.log_dir = curr_time.strftime("%Y_%m_%d__%H_%M_%S")
        os.mkdir(args.log_dir)
        print("Logging to directory: ", args.log_dir)

    # Create separate folders for each camera
    os.chdir(args.log_dir)
    for cam in camera_params.keys():
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
            processes = [centre_process, side_process]
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
            centre_process = multiprocessing.Process(target=centre_image_logging_worker, args=(camera_params,))
            side_process = multiprocessing.Process(target=side_image_logging_worker, args=(camera_params, buffer))
            processes = [centre_process, side_process]

            for process in processes:
                process.start()

            # Start the consumer pool that writes the side images to file
            if args.log_data:
                pool_size = args.jpg_writer_pool_size
                consumer_pool = [multiprocessing.Process(target=side_image_logging_writer, args=(buffer,)) for _ in range(pool_size)]
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