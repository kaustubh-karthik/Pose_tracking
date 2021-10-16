import cv2
import mediapipe as mp
import numpy as np
import time
class pose_detector():
    def __init__(self, mode = False, max_hands = 2, detection_conf = 0.5, track_conf = 0.5):
        # Do i need variables? -- Check for static
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.prev_time = 0


    def find_lms(self, img, draw = True, lm_id = None):
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_img)

        lm_list = []

        if results.pose_landmarks:

            for id, landmark in enumerate(results.pose_landmarks.landmark):

                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                lm_list.append([id, cx, cy])

                if draw:
                    self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0))

        if lm_id != None:
            print(lm_list[lm_id])

        return img, lm_list

    def display_img(self, img, fps = True):

        if fps:
            curr_time = time.time()
            fps = 1/(curr_time - self.prev_time)
            prev_time = curr_time


        cv2.putText(
            img,
            text = str(int(fps)),
            org = (10, 70),
            fontFace = cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 3,
            color = (150, 150, 150),
            thickness = 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def main():

    detector = pose_detector()

    cap = cv2.VideoCapture('pose_tracking/Pose_videos/pexels-rodnae-productions-7187087.mp4') # Check for error

    while True:
        success, img = cap.read()

        img, lm_list = detector.find_lms(img, lm_id=4)

        detector.display_img(img)

if __name__ == '__main__':
    main()
