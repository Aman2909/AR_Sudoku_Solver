'''Applied Gaussian Blur and Adaptive Threshold Gaussian'''

import cv2,time
video = cv2.VideoCapture(0)
#tic = time.time()
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    #cv2.imshow("Gray", gray)
    # cv2.imshow("Blur", blur)

    # img_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("Thresh", img_thresh)

    img_thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    cv2.imshow("Thresh2", img_thresh2)

    key = cv2.waitKey(1)
    if key == ord('q'):
        #toc = time.time()
        break
video.release()
#print(toc - tic)
cv2.destroyAllWindows()