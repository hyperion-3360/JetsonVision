
import sys
from threading import Thread
from dalsa import Camera
from stream_common import FfmpegEncoderX264, read_encoded


#
#
# Get the common support code for the GigE-V Framework for Linux
# (Change this if directory structure is changed).
import os
sys.path.append(os.path.dirname(__file__) + "./gigev_common")
import pygigev  # includeded in ../gigev_common, DO NOT install from pip


def stream_and_feed(camera: Camera, encoder_process):
    """
    Stream a camera feed and send it to the h.264 encoder as fast as possible
    Run this in a thread
    """
    camera.open()
    camera.setup(buffer_count=1)

    while(True):
        for frame in camera.read():
            FfmpegEncoderX264.feed_frame(encoder_process, frame)

def send_over_net(encoder_process):
    for frame in read_encoded(encoder_process, 1920, 1080):
        ...

if __name__ == "__main__":
    # Initialize the API
    pygigev.GevApiInitialize()

    camera = Camera(index=0)
    encoder = FfmpegEncoderX264.start_ffmpeg()

    streamer_t = Thread(target=stream_and_feed, args=[camera, encoder])
    reader_t = Thread(target=send_over_net, args=[encoder])

    streamer_t.run()
    reader_t.run()

    streamer_t.join()
    reader_t.join()

    # TODO: somehow stop the threads and cleanup
    FfmpegEncoderX264.release(encoder)
    camera.release()

    # Uninitialize
    pygigev.GevApiUninitialize()

