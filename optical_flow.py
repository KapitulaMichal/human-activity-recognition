file = './all_actions/person01_running_d1_uncomp.avi'

import cv2
import numpy as np

cap = cv2.VideoCapture(file)
ret, frame1 = cap.read()
print(frame1)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

for _ in range(14):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    prvs = next

ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
cv2.imwrite('frame2.png', next)


# next = cv2.resize(next,(int(next.shape[1]*imgScale), int(next.shape[1]*imgScale)))
# prvs = cv2.resize(prvs,(int(prvs.shape[1]*imgScale), int(prvs.shape[1]*imgScale)))

flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame2',bgr)
k = cv2.waitKey(30) & 0xff
print(bgr.shape)
imgScale = 0.5
#bgr = cv2.resize(bgr,(int(bgr.shape[1]*imgScale), int(bgr.shape[0]*imgScale)))
print(bgr.shape)
#bgr = np.reshape(bgr, (np.product(bgr.shape),))
print(bgr)

a = np.array([bgr])
print(a)
a = np.append(a,[bgr], axis=0)
print(a)
print(type(a))
print(a.shape)

cv2.imwrite('opticalhsv.png',bgr)


'''
import cv2
import numpy as np


cap = cv2.VideoCapture(file)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame2',bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
cap.release()
cv2.destroyAllWindows()

'''