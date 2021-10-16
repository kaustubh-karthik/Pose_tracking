import cv2
import mediapipe as mp
import time


mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


video = cv2.VideoCapture('/Users/kaustubhkarthik/Documents/Tensorflow-projects/pose_tracking/Pose_videos/pexels-artem-podrez-8992774.mp4')

prev_time = 0

while True:
    success, img = video.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_img)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
