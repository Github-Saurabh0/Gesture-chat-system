import cv2

from mediapipe.python.solutions import hands
from mediapipe.python.solutions import drawing_utils


class GestureDetector:

    def __init__(self):

        self.mp_hands = hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_draw = drawing_utils

    def detect_gesture(self, hand_landmarks):

        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other Fingers
        for tip in tips[1:]:

            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Gesture Logic
        if fingers == [0, 1, 0, 0, 0]:
            return "HELLO"

        elif fingers == [0, 1, 1, 0, 0]:
            return "HI"

        elif fingers == [1, 1, 1, 1, 1]:
            return "STOP"

        elif fingers == [0, 1, 1, 1, 1]:
            return "YES"

        return "UNKNOWN"

    def process_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        gesture = "No Hand Detected"

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                gesture = self.detect_gesture(hand_landmarks)

        return frame, gesture