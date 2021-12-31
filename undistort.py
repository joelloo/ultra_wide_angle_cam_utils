import cv2 as cv
import numpy as np
import time

# calib = np.load("calibration_params.npz")
# calib = np.load("wide_angle_calibration.npz")
calib = np.load("right_calibration.npz")
mtx = calib['camera_matrix']
dist = calib['distortion']
print(mtx)
print(dist)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print( "Error opening camera" )
    exit(0)
else:
    while True:
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            # w = 910
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            break

print("=== Optimal camera matrix ===")
print(newcameramtx)
print("=== ROI ===")
print(roi)

map1, map2 = cv.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (w, h), cv.CV_16SC2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Dropped image")
        time.sleep(0.025)
        continue

    frame = frame[:, 1010:, :]
    # dst = cv.undistort(frame, mtx, dist, None, mtx)
    #dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    dst = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    cv.imshow("Raw image stream", frame)

    # x, y, w, h = roi
    #cv.imshow("Undistorted", dst[y:y+h, x:x+w])
    cv.imshow("Undistorted", dst)
    key = cv.waitKey(50)