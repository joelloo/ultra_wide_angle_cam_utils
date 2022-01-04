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
    parser.add_argument("--target_arm64", type=bool, help="uses binaries targeted at arm64 if true: e.g. uses prepackaged arm64 ffmpeg, ffprobe binaries etc.",
        default=False)
    args = parser.parse_args()

    arm64_exe_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ffmpeg-4.4.1-arm64-static") if args.target_arm64 else None
    if arm64_exe_path is not None and not os.path.exists(arm64_exe_path):
        print("Unpacking packaged ffmpeg")
        sp.run(["tar", "-xvf", "ffmpeg-release-arm64-static.tar.xz"])
    ffmpeg_path = "ffmpeg"
    ffprobe_path = "ffprobe"
    if arm64_exe_path is not None:
        ffmpeg_path = arm64_exe_path + "/ffmpeg"
        ffprobe_path = arm64_exe_path + "/ffprobe"

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
        convert_process = sp.Popen(shlex.split(f'{ffmpeg_path} -i {vid_file} -f image2 -frame_pts true %06d.jpg'))
        ret = convert_process.wait()

        print("Checking mp4 presentation timestamps")
        ts_json_out = sp.Popen(shlex.split(f'{ffprobe_path} {vid_file} -v quiet -select_streams v -of json -show_entries frame=best_effort_timestamp_time'),
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