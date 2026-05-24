import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    # ROI Box
    cv2.rectangle(frame, (150, 100), (350, 300), (0, 255, 0), 2)
    roi = frame[100:300, 150:350]

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin Color Range
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Blur
    kernel = np.ones((3,3), np.uint8)

    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    gesture = "No Hand"

    if contours:

        cnt = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(cnt)

        if area > 5000:

            hull = cv2.convexHull(cnt)

            hull_area = cv2.contourArea(hull)

            area_ratio = ((hull_area - area) / area) * 100

            defects = cv2.convexityDefects(
                cnt,
                cv2.convexHull(cnt, returnPoints=False)
            )

            fingers = 0

            if defects is not None:

                for i in range(defects.shape[0]):

                    s, e, f, d = defects[i, 0]

                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))

                    angle = np.arccos(
                        (b**2 + c**2 - a**2) / (2*b*c)
                    ) * 57

                    if angle <= 90:
                        fingers += 1

                        cv2.circle(
                            roi,
                            far,
                            5,
                            [0, 0, 255],
                            -1
                        )

            fingers += 1

            # Gesture Mapping
            if fingers == 1:
                gesture = "HELLO"

            elif fingers == 2:
                gesture = "HI"

            elif fingers == 3:
                gesture = "YES"

            elif fingers >= 4:
                gesture = "STOP"

    cv2.putText(
        frame,
        f'Gesture: {gesture}',
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Gesture Chat System", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()