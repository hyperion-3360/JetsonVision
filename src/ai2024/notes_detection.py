
import os

import cv2
import numpy as np


def ref_circle(radius):
    radius = round(radius)
    canva = np.zeros((radius*2, radius*2), dtype=np.uint8)
    canva = cv2.circle(canva, (radius, radius), radius, 255, 1)

    return np.asarray(list(zip(*np.where(canva == 255))))

def detect_notes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold of orange in HSV space
    lower_orange = np.array([0, 120, 200])
    upper_orange = np.array([25, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    color_masked = cv2.bitwise_and(frame, frame, mask = mask)

    _, gray, _ = cv2.split(color_masked)

    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=2)
    # gray[np.where(gray > 0)] = 255

    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # gray = cv2.GaussianBlur( gray, (9, 9), 2, 2 )

    cv2.imshow("masked", color_masked)
    cv2.imshow("gray", gray)

    cv2.waitKey(0)

if __name__ == "__main__":
    # for file in os.listdir('notes'):
    #     f = cv2.imread(f'notes/{file}')
    #     detect_notes(f)

    f = cv2.imread(f'notes/test.png')
    detect_notes(f)