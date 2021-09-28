import cv2 as cv
import numpy as np

# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 100, 100])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160,100,100])
upper2 = np.array([179,255,255])

cam = cv.VideoCapture(0)

while(True):
    ret,img = cam.read()

    if img is None:
        break

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_mask = cv.inRange(img_hsv, lower1, upper1)
    upper_mask = cv.inRange(img_hsv, lower2, upper2)
    
    frame_threshold = lower_mask + upper_mask;
    frame_threshold = cv.GaussianBlur(frame_threshold, (7,7), 2,None,2)

    temp_array = []

    width = img.shape[1] # width
    chunk_size = width//10
    chunk_start = 0
    chunk_end = chunk_size
    for i in range(10):
        temp_array.append(np.count_nonzero(frame_threshold[:,chunk_start:chunk_end] > 230)//1000)
        chunk_start = chunk_end
        chunk_end += chunk_size


    print(temp_array)
    cv.imshow("frame", img)
    cv.imshow("filter", frame_threshold)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()