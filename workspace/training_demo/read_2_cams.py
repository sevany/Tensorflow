# import cv2
# import numpy as np
# # import imutils
# # from imutils.video import VideoStream


# # # Declare camera/video input
# cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(2)
# # cap1 = VideoStream(src=0).start()
# # cap2 = VideoStream(src=1).start()


# # while cap1.isOpened() or cap2.isOpened():
# while True:

#     okay1  , frame1 = cap1.read()
#     okay2 , frame2 = cap2.read()

#     if okay1:
#         hsv1 = cv2.cvtColor(frame1 , cv2.COLOR_BGR2HSV)
#         cv2.imshow('fake' , hsv1)

#     if okay2:
#         hsv2 = cv2.cvtColor(frame2 , cv2.COLOR_BGR2HSV)
#         cv2.imshow('real' , hsv2)

#     if not okay1 or not okay2:
#     # if not okay1:
#         print('Cant read the video , Exit!')
#         break

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
    
#     cv2.waitKey(25)

# cap1.release()
# cap2.release()
# cv2.destroyAllWindows()

import numpy as np
import cv2

video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(2)

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', frame0)

    if (ret1):
        # Display the resulting frame
        cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()