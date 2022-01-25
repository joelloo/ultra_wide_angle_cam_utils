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

### Example
A concrete example of the steps to be taken when running the driver onboard the Spot's Xavier are listed below:
* Make sure that the Xavier's power settings are set to `30W ALL`.
* Check the video device IDs of the cameras. The cameras will be registered as video devices at the directory `/dev/videoX`, with different `X`. As there may be multiple devices registered, it is recommended that you run the `check_camera.py` script with the following arguments to find out which ID belongs to the centre camera and side camera respectively:
```
python3 check_camera.py --cam_id X --force_v4l2 True
```
where `X` is the video device ID number you are currently testing. **Please verify the video device IDs for both centre and side cameras every time you run the driver after either 1) restarting the computer or 2) replugging the USB cables. It is possible that the IDs can change in these cases.**
* Once you have verified the device IDs for the centre and side cameras respectively, you can run the driver with the following instructions:
```
python3 stream_camera.py --centre_id [fill in id no.] --side_id [fill in id no.] --force_v4l2 True
```
This will log images from the camera **after passing them through OpenCV's undistortion routines**. If you wish to log distorted images directly from the camera (i.e. raw data), you may add in the following flag:
```
python3 stream_camera.py --centre_id [fill in id no.] --side_id [fill in id no.] --force_v4l2 True --log_distorted 0
```
* Logging intention/control data can be done the usual way, by using the Record panel in Isaac's Websight. The driver records data in **Unix epoch time** using the system wall clock, whereas the intention/control data is logged using **Isaac's app clock time**. The Unix epoch time differs from Isaac's app clock time by a constant offset, and this offset is written to `/data/ultra_wide_angle_cam_utils/app_clock_offset.txt` **every time the INet system is started up**. When logging camera data, please do remember to save the offset with your recorded logs, **before restarting INet**.
* Please do remember to switch the power mode back to **MAXN** once you are done logging. Otherwise the Spot's performance may be poor during testing.