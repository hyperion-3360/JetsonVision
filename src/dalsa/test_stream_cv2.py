import subprocess
from threading import Thread
import cv2
import sys

import numpy as np

from stream_common import FfmpegEncoderX264, read_encoded


def test_stream_and_feed(encoder_process: FfmpegEncoderX264.ProcessInfo):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FfmpegEncoderX264.feed_frame(encoder_process, frame)


def send_over_net(encoder_process: FfmpegEncoderX264.ProcessInfo):

    # gstreamer = subprocess.Popen(['gst-launch-1.0', 'videotestsrc', '!', 'vtenc_h264', '!', 'rtph264pay', 'config-interval=10', 'pt=96', '!', 'udpsink', 'host=127.0.0.1', 'port=5000'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    # gstreamer = subprocess.Popen(['/Users/andlat/Documents/FIRST2024/JetsonVision/src/gstreamer_scripts/test.sh'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)

    # Read directly from pipe
    cap = cv2.VideoCapture('pipe:{}'.format(encoder_process.raw.stdout.fileno()))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('H', '2', '6', '4'))

    gstreamer_str = "appsrc ! vtenc_h264 ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000"
    out = cv2.VideoWriter(gstreamer_str, 0, 30, (1280, 720))

    while(True):
        r, f = cap.read()
        if(r):
            out.write(f)
            # data = f.astype(np.uint8).tobytes()
            # gstreamer.stdin.write(data)
            # gstreamer.communicate(f)

            # cv2.imshow("title", f)
            # cv2.waitKey(1)

    # for frame in read_encoded(encoder_process):
    #     # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    #     cv2.imshow("title", frame)
    #     cv2.waitKey(1)


if __name__ == "__main__":
    encoder: FfmpegEncoderX264.ProcessInfo = FfmpegEncoderX264.start_ffmpeg(1280, 720)

    # test_stream_and_feed(encoder)
    streamer_t = Thread(target=test_stream_and_feed, args=[encoder])

    streamer_t.start()

    send_over_net(encoder)

    # TODO: somehow stop the threads and cleanup
    FfmpegEncoderX264.release(encoder)
    streamer_t.join()