from threading import Thread
import cv2

from stream_common import FfmpegEncoderX264, read_encoded


def test_stream_and_feed(encoder_process: FfmpegEncoderX264.ProcessInfo):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FfmpegEncoderX264.feed_frame(encoder_process, frame)


def send_over_net(encoder_process: FfmpegEncoderX264.ProcessInfo):
    for frame in read_encoded(encoder_process):
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        cv2.imshow("title", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    encoder: FfmpegEncoderX264.ProcessInfo = FfmpegEncoderX264.start_ffmpeg(1280, 720)

    # test_stream_and_feed(encoder)
    streamer_t = Thread(target=test_stream_and_feed, args=[encoder])

    streamer_t.start()
    send_over_net(encoder)

    # TODO: somehow stop the threads and cleanup
    FfmpegEncoderX264.release(encoder)
    streamer_t.join()