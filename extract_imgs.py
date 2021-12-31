import subprocess as sp
import shlex
import argparse
import os
import json
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="specify log directory",
        default=None)
    args = parser.parse_args()

    if args.log_dir is None:
        print("No log directory specified!")
        exit(0)

    os.chdir(args.log_dir)
    dirs = os.listdir()

    for dir in dirs:
        print("Processing ", dir)
        os.chdir(dir)
        print("Converting mp4 to jpg")
        vid_file = dir + ".mp4"
        convert_process = sp.Popen(shlex.split(f'ffmpeg -i {vid_file} -f image2 -frame_pts true %06d.jpg'))
        ret = convert_process.wait()

        print("Checking mp4 presentation timestamps")
        ts_json_out = sp.Popen(shlex.split(f'ffprobe {vid_file} -v quiet -select_streams v -of json -show_entries frame=best_effort_timestamp_time'),
            stdout=sp.PIPE)
        ts_json = json.loads(ts_json_out.stdout.read())

        if 'frames' not in ts_json:
            print("Invalid timestamp response received from ffprobe. Does the video contain any valid frames?")
            exit(0)
        timestamps = []
        for frame in ts_json['frames']:
            ts = float(frame['best_effort_timestamp_time'])
            timestamps.append(ts)

        print("Rename images by timestamps")
        imgs = glob.glob("*.jpg")
        assert(len(imgs) == len(timestamps))

        img_indices = sorted([int(img[:-4]) for img in imgs])
        for img_idx, ts in zip(img_indices, timestamps):
            ts_int = int(round(ts * 1000))
            os.rename(('%06d' % img_idx)  + '.jpg', str(ts_int) + ".jpg")

        print("Completed ", dir)
        os.chdir("..")