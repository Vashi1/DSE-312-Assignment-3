import cv2
import numpy as np

cap = cv2.VideoCapture("Traffic_intersection.mp4")  # Replace with your video source

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=3000, qualityLevel=0.3, minDistance=3)

# Define the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Corrected this line
out = cv2.VideoWriter('output_video.avi', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)

    # Select good points
    good_new = new_pts[status == 1]
    good_prev = prev_pts[status == 1]

    # Draw tracks on the frame
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        a, b = new.ravel()
        c, d = prev.ravel()
        frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Update the previous frame and points
    prev_gray = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    # Write the frame with optical flow to the output video
    out.write(frame)

    # Display the video frame with optical flow tracks
    cv2.imshow("Optical Flow (Lucas-Kanade)", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the VideoWriter and capture objects
out.release()
cap.release()
