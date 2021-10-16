import cv2
from pose_tracking_utils import pose_detector

detector = pose_detector()

cap = cv2.VideoCapture('pose_tracking/Pose_videos/pexels-rodnae-productions-7187087.mp4') # Check for error

while True:
    success, img = cap.read()

    img, lm_list = detector.find_lms(img, lm_id=4)

    detector.display_img(img)
