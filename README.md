# Ultra-wide-angle camera utilities
## Description
This repository comprises a few scripts to allow processing of data from the ultra-wide-angle, dual-sensor camera. The scripts are:
* stream_camera.py: Allows either streaming or logging of the data received from the camera. Allows data to be logged directly as jpgs, or as HEVC/h.265 encoded mp4 files. (Currently the mp4 logging function only works with x86-based FFMPEG binaries.)
* extract_imgs.py: Running this script converts data logged as HEVC/h.265 encoded mp4 files into jpgs.
* check_camera.py: Simple script that streams the video feed from a specified camera ID for visualisation, and checks the FPS of the feed.
* calibrate.py: Not intended for general use. Tool used to calibrate the cameras. The calibrations for the current camera are also provided in the repository as .npz files.
* undistort.py: Not intended for general use. Tool used to test calibration by visualising the undistorted images.
## Installation
The tools are written entirely in Python, and have been tested in a conda environment using Python 3.8. The conda environment used for testing is specified in the provided environment.yml. It is recommended that the user create a conda environment directly from this environment.yml and run the scripts inside the environment.

If logging to video is needed, the repository contains packaged ffmpeg and ffprobe binaries that will be unpacked and called instead of the user's system ffmpeg and ffprobe. The compiled binaries are obtained from the following: https://superuser.com/questions/1302753/ffmpeg-unrecognized-option-crf-error-splitting-the-argument-list-option-not
## Usage
Tested on macOS and Linux. On x86 architectures, data can be logged using the following command
`python3 stream_camera.py --centre_id [centre cam id] --side_id [side cam id] --log_data True`

On ARM architectures, it is necessary to force OpenCV to use the correct driver. The following command should be used
`python3 stream_camera.py --centre_id [centre cam id] --side_id [side cam id] --log_data True --force_v4l2 True`

Using the `--visualise` flag allows the camera streams to be displayed in realtime. It is also possible to log as HEVC-encoded mp4 videos using the `--log_video` flag. However, there are currently some issues using this option on ARM architectures.
