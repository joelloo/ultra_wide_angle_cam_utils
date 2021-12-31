import numpy as np
import cv2 as cv
import time

# Camera from /dev/videoX
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print( "Error opening camera" )
    exit(0)

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp *= 0.0214 # Multiplicative factor for 1cm chessboards scaled to A4 size

print("=== Object points ===")
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Dropped image")
        time.sleep(0.025)
        continue
    
    frame = frame[:, 1010:, :]
    cv.imshow("Camera view", frame)
    key = cv.waitKey(50)
    if key == ord('s'):
        print("Capturing image...")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            print("Displaying detections...")
            cv.drawChessboardCorners(frame, (9,6), corners2, ret)
            cv.imshow('Chessboard', frame)
            cv.waitKey(0)
    elif key == ord('q'):
        break

# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# if ret:
#     print("=== Camera matrix ===")
#     print(mtx)
#     print("=== Distortion ===")
#     print(dist)
#     print("=== Rotation ===")
#     print(rvecs)
#     print("=== Translation ===")
#     print(tvecs)

#     savevals = {"camera_matrix":mtx, "distortion":dist, "R":rvecs, "t":tvecs}

#     print('Saved to calibration_params.npz')
#     np.savez("calibration_params.npz", **savevals)

objpoints = np.expand_dims(np.array(objpoints, dtype=np.float32), -2)
imgpoints = np.array(imgpoints, dtype=np.float32)

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW
shape = frame.shape[:-1]
ret, _, _, _, _ = cv.fisheye.calibrate(
    objpoints, imgpoints, shape[::-1], K, D, rvecs, tvecs, calibration_flags, criteria)
if ret:
    print("=== K ===")
    print(K)
    print("=== D ===")
    print(D)
    print("=== Rotation ===")
    print(rvecs)
    print("=== Translation ===")
    print(tvecs)

    savevals = {"camera_matrix":K, "distortion":D, "R":rvecs, "t":tvecs}

    print("Saved to calibration_params.npz")
    np.savez("calibration_params.npz", **savevals)