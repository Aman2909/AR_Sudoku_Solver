'''
0. FPS has been reduced
1. Contours detected
2. Biggest Contour is our Sudoko puzzle
'''

import cv2,math
import numpy as np
video = cv2.VideoCapture(0)

while True:
    num_frames = 8
    for i in range(num_frames):
        check, frame = video.read() #Slows the frame rate
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("Blur", blur)

    img_thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("Thresh2", img_thresh2)

    frame2 = np.copy(frame)
    contours, hierarchy = cv2.findContours(img_thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("Contours",frame)

    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    print(biggest)
    cv2.drawContours(frame2,biggest, -1, (0, 0, 255), -1)
    if biggest is not None:
        cv2.fillPoly(frame2, pts=[biggest], color=(255, 255, 255))
    cv2.imshow("Biggest",frame2)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()





print("Top_Right : ",Top_Right)
print("Bottom_left : ",Bottom_Left)
print("Bottom_Right : ",Bottom_Right)
print("Top_left : ", Top_Left)