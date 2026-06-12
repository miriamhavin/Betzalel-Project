import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. reduce noise slightly (but keep structure)
    blur = cv2.GaussianBlur(frame, (5,5), 0)

    # 2. BOOST COLORS HARD (RGB amplification style)
    bright = cv2.convertScaleAbs(blur, alpha=1.6, beta=30)

    # 3. posterize (remove details)
    bright = (bright // 40) * 40

    # 4. edges (ONLY structure)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 140)

    # thicken edges so they dominate
    edges = cv2.dilate(edges, None, iterations=2)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 5. overlay edges on flat colors
    cartoon = cv2.max(bright, edges)

    cv2.imshow("bright edges cartoon", cartoon)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()