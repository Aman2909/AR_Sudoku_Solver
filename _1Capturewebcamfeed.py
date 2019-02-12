'''
Step 1: Import cv2
Step 2: Create object for VideoCapture()
Step 3: Video is a sequence of images. Therefore looping an image will "stream" our video.
        Start a while loop that is true always.
Step 4: Run the read function on the object. It returns two things.
        1. If everything is okay it stores "TRUE" in the first return value
        2. In the second parameter it stores the intensity values for all the pixels in the form of an array

Step 5: (Optional) We convert the frame into grayscale
Step 6: Display by imshow()
Step 7: Set a breaking condition. (Here) if the input is "q" then it stops streaming
Step 8: Release the video object
step 9: Destroy all the windows
Step 10:(Optional) To calulate the lenght of time streaming was done, use time object (toc-tic) concept
'''

import cv2,time
video = cv2.VideoCapture(0)
print("Width : " +str(video.get(3)))
print("Height : " +str(video.get(4)))
tic = time.time()
while True:
    check, frame = video.read()
    #print(check)
    #print(frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #5
    #cv2.imshow("Captured",frame)

    cv2.imshow("Captured",gray)
    #cv2.waitKey(0)

    key = cv2.waitKey(1)
    if key == ord('q'):
        toc=time.time()
        break
#3
video.release()
print(toc-tic)
cv2.destroyAllWindows()

