
import argparse
import os
from pathlib import Path

import cv2

parser = argparse.ArgumentParser(
                    prog='Extract Frames',
                    description='Extract specific frames from a video. Click on \'s\' to save a frame or on the spacebar to continue to the next frame'
        )

parser.add_argument('videopath', type=Path)
parser.add_argument('-d', '--destination', dest='dest', type=Path)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    src = args['videopath']
    assert src is not None

    src = Path(src)

    dest = args['dest']
    if dest is None:
        dest = src.parent / src.stem

    dest.mkdir(parents=True, exist_ok=True)
    fi = len(os.listdir(dest))

    cap = cv2.VideoCapture(str(src))

    while True:
        r,f  = cap.read()

        if not r:
            break

        cv2.imshow("Video Capture", f)
        k = cv2.waitKey(0)
        if k == ord('s'):
            cv2.imwrite(str(dest/f'calibration_{fi}.png'), f)
            fi += 1
        elif k == ord('q'):
            break

    cap.release()

