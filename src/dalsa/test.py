# #! /usr/bin/python3

# # /IT License
# # Copyright (c) 2019,2020 JetsonHacks
# # See license
# # A very simple code snippet
# # Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# # NVIDIA Jetson Nano Developer Kit (Rev B01) using OpenCV
# # Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# # This script will open a window and place the camera stream from each camera in a window
# # arranged horizontally.
# # The camera streams are each read in their own thread, as when done sequentially there
# # is a noticeable lag
# # For better performance, the next step would be to experiment with having the window display
# # in a separate thread

# import cv2
# import threading
# import numpy as np

# # gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# # Flip the image by setting the flip_method (most common values: 0 and 2)
# # display_width and display_height determine the size of each camera pane in the window on the screen

# cam = None

# class CSI_Camera:

#     def __init__ (self) :
#         # Initialize instance variables
#         # OpenCV video capture element
#         self.video_capture = None
#         # The last captured image from the camera
#         self.frame = None
#         self.grabbed = False
#         # The thread where the video capture runs
#         self.read_thread = None
#         self.read_lock = threading.Lock()
#         self.running = False


#     def open(self, gstreamer_pipeline_string):
#         try:
#             self.video_capture = cv2.VideoCapture(
#                 gstreamer_pipeline_string, cv2.CAP_GSTREAMER
#             )
            
#         except RuntimeError:
#             self.video_capture = None
#             print("Unable to open camera")
#             print("Pipeline: " + gstreamer_pipeline_string)
#             return
#         # Grab the first frame to start the video capturing
#         self.grabbed, self.frame = self.video_capture.read()

#     def start(self):
#         if self.running:
#             print('Video capturing is already running')
#             return None
#         # create a thread to read the camera image
#         if self.video_capture != None:
#             self.running=True
#             self.read_thread = threading.Thread(target=self.updateCamera)
#             self.read_thread.start()
#         return self

#     def stop(self):
#         self.running=False
#         self.read_thread.join()

#     def updateCamera(self):
#         # This is the thread to read images from the camera
#         while self.running:
#             try:
#                 grabbed, frame = self.video_capture.read()
#                 with self.read_lock:
#                     self.grabbed=grabbed
#                     self.frame=frame
#             except RuntimeError:
#                 print("Could not read image from camera")
#         # FIX ME - stop and cleanup thread
#         # Something bad happened
        

#     def read(self):
#         with self.read_lock:
#             frame = self.frame.copy()
#             grabbed=self.grabbed
#         return grabbed, frame

#     def release(self):
#         if self.video_capture != None:
#             self.video_capture.release()
#             self.video_capture = None
#         # Now kill the thread
#         if self.read_thread != None:
#             self.read_thread.join()


# # Currently there are setting frame rate on CSI Camera on Nano through gstreamer
# # Here we directly select sensor_mode 3 (1280x720, 59.9999 fps)
# def gstreamer_pipeline():
#     return (
#         "nvarguscamerasrc sensor-id=1 sensor-mode=3 ! "
#         "video/x-raw(memory:NVMM), "
#         "  width=1280, "
#         "  height=720, "
#         "  format=NV12, "
#         "  framerate=60/1 ! "
#         "nvvidconv flip-method=0 ! "
#         "videoconvert ! "
#         "video/x-raw, format=BGR ! "
#         " appsink"
#     )

# def gstreamer_pipeline_out():
#     return (
#         "appsrc ! "
#         "video/x-raw, format=BGR ! "
#         "queue ! "
#         "videoconvert ! "
#         "video/x-raw, format=BGRx ! "
#         "nvvidconv ! "
#         "omxh264enc ! "
#         "video/x-h264, stream-format=byte-stream ! "
#         "h264parse ! "
#         "rtph264pay pt=96 config-interval=1 ! "
#         "udpsink host=127.0.0.1 port=5000"
#     )

# def start_cameras():
#     cam = cv2.VideoCapture(0)

#     if (not cam.isOpened()):
#         print("Unable to open any cameras")
#         SystemExit(0)

#     out = cv2.VideoWriter(gstreamer_pipeline_out(), 0, 60, (1280,720))

#     while not out.isOpened():
#       print('VideoWriter not opened')
#       SystemExit(0)

#     while True :
#         _ , frame=cam.read()
#         # img = cv2.blur(frame,(3,3))
#         #img = cv2.medianBlur(img,9)
#         # out.write(img)
#         cv2.imshow("tt",  frame)
#         cv2.waitKey(1)

#     cam.stop()
#     cam.release()

# if __name__ == "__main__":
#     start_cameras()


import sys
import cv2

def read_cam():
    cap = cv2.VideoCapture(0)

    w = 1920
    h = 800
    fps = 30
    print('Src opened, %dx%d @ %d fps' % (w, h, fps))

    gst_out = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000"
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))
    if not out.isOpened():
        print("Failed to open output")
        exit()

    if cap.isOpened():
        while True:
            ret_val, img = cap.read()
            # cv2.imshow('ttt', img)
            if not ret_val:
                break

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = cv2.flip(img, 0)
            print(img.shape)
            out.write(img)
            # cv2.waitKey(1)
    else:
     print("pipeline open failed")

    print("successfully exit")
    cap.release()
    out.release()


if __name__ == '__main__':
    read_cam()