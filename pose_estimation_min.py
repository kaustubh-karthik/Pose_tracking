import cv2
import mediapipe as mp
import time


mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


video = cv2.VideoCapture('Pose_videos/pexels-kampus-production-8636818.mp4')

prev_time = 0

while True:
    success, img = video.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_img)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, landmark in enumerate(results.pose_landmarks.landmark):

            height, width, channel = img.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)

            print(f'id: {id}\nlandmark: {landmark}')

            cv2.circle(img, (cx, cy), 10, (255, 0, 0))

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
