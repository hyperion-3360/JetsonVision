gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw, framerate=30/1,width=640,height=480 ! videoconvert ! jpegenc ! rtpjpegpay ! udpsink host=192.168.0.219 port=5000
