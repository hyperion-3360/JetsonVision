#!/bin/bash

# linux
# gst-launch-1.0 videotestsrc ! openh264enc ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
# macos
# gst-launch-1.0 videotestsrc ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
# gst-launch-1.0 fdsrc fd=0  ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
# opencv video writer: appsrc  ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000

v4l2src device=/dev/video1 ! video/x-raw, width=1280, height=720 ! nvvidconv ! video/x-raw, format=I420 ! x264enc ! video/x-h264, stream-format=byte-stream ! h264parse ! rtph264pay name=pay0 pt=96

gst-launch-1.0 rtspsrc latency=200 location="rtsp://10.117.17.112:8554/test" ! rtph264depay ! h264parse ! nvh264dec ! autovideosink


gst-launch-1.0 -vvv \
tcpclientsrc host=$IP_ADDRESS port=5000 \
! tsdemux \
! h264parse ! omxh264dec ! videoconvert \
! xvimagesink sync=false

gst-launch-1.0 tcpclientsrc host=<Jetson_IP> port=4953 ! queue ! matroskademux ! h264parse ! avdec_h264 ! queue ! videoconvert ! fpsdisplaysink text-overlay=0 video-sink=fakesink -v


######## WORKING MAGIC FORMULA ##########

# CLIENT:
gst-launch-1.0 tcpclientsrc host=10.117.17.112 port=4953 ! queue ! matroskademux ! h264parse ! avdec_h264 ! queue ! videoconvert ! osxvideosink

# SERVER
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1,width=640,height=480 ! queue ! nvvidconv ! nvv4l2h264enc insert-vui=1 insert-sps-pps=1 idrinterval=15 ! h264parse ! matroskamux streamable=1 ! tcpserversink host=10.117.17.112 port=4953

#### UDP ?????

# SERVER
gst-launch-1.0 videotestsrc is-live=1 ! video/x-raw,width=1280,height=720 ! timeoverlay valignment=4 halignment=1 ! nvvidconv ! 'video/x-raw(memory:NVMM),width=1280,height=720' ! tee name=t ! nvv4l2h264enc insert-sps-pps=1 idrinterval=15 ! h264parse ! rtph264pay ! udpsink host=192.168.50.101 port=4953 sync=0 t. ! queue ! nvegltransform ! nveglglessink sync=0

# CLIENT
gst-launch-1.0 udpsrc port=4953 ! 'application/x-rtp,encoding-name=H264,payload=96' ! rtph264depay ! avdec_h264 ! osxvideosink sync=0


### LOCAL UDP WORKING
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=1280,height=720 ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000

gst-launch-1.0 -v udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse  ! avdec_h264 ! videoconvert ! autovideosink


gst-launch-1.0 videotestsrc ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000


### MacOS --> Jetson

#Mac server:
gst-launch-1.0 avfvideosrc ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay config-interval=10 pt=96 ! udpsink host=10.117.17.112 port=5000

# Jetson client:
gst-launch-1.0 -v udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse  ! avdec_h264 ! videoconvert ! autovideosink


### Jetson robot ---> Mac

# Server (jetson)
# P.S. Select the correct camera
gst-launch-1.0 v4l2src device=/dev/video0 ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay config-interval=10 pt=96 ! udpsink host=10.33.60.212 port=5000

# Client (mac)
gst-launch-1.0 -v udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse  ! avdec_h264 ! videoconvert ! autovideosink

