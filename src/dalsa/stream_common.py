from typing import Generator, Optional
import ffmpeg
import numpy as np
from time import sleep

class FfmpegEncoderX264:
    class ProcessInfo:
        def __init__(self, process, w: int, h: int):
            self.raw = process
            self.resolution = (w, h)

        def raise_if_unstarted(self):
            if self.raw is None:
                raise Exception("ffmpeg process not started. Satrt it with FfmpegEncoderX264.start_ffmpeg before calling this method")

    @staticmethod
    def start_ffmpeg(width: int, height: int):
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb32', s=f'{width}x{height}')
            .video
            .output('pipe:', format="h264", vcodec='libx264', pix_fmt='yuv444p')
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )

        return FfmpegEncoderX264.ProcessInfo(process, width, height)

    @staticmethod
    def feed_frame(process: ProcessInfo, frame: np.ndarray):
        """
        Send the frame to ffmpeg through stdin
        """
        process.raise_if_unstarted()

        data = frame.astype(np.uint32).tobytes()
        process.raw.stdin.write(data)

    @staticmethod
    def read_encoded(process: ProcessInfo) -> Optional[np.ndarray]:
        process.raise_if_unstarted()

        # Read raw video frame from stdout as bytes array.
        w, h = process.resolution
        in_bytes = process.raw.stdout.read(w * h * 3)
        if not in_bytes:
            return None

        # transform the byte read into a numpy array
        return np.frombuffer(in_bytes, np.uint8).reshape([h, w, 3])

    @staticmethod
    def release(process: ProcessInfo):
        process = process.raw
        if process is not None:
            process.stdin.close()


# class NetworkSender:
#     def __init__(self, client_ip: str):
#         self.port = 5555
#         self.client_ip = client_ip


def read_encoded(encoder_process: FfmpegEncoderX264.ProcessInfo) -> Generator[np.ndarray, None, None]:
    """
    Read the ffmpeg encoder's output as fast as possible and yield the encoded images
    Run this in a thread
    """
    while(True):
        frame = FfmpegEncoderX264.read_encoded(encoder_process)

        if frame is not None:
            yield frame

        else:
            sleep(0.01) # sleep 10ms if no