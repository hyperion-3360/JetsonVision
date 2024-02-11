#!/bin/bash

# linux
# gst-launch-1.0 videotestsrc ! openh264enc ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
# macos
# gst-launch-1.0 videotestsrc ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
# gst-launch-1.0 fdsrc fd=0  ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
# opencv video writer: appsrc  ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000