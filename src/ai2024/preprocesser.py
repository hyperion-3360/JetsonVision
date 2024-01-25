import cv2
import numpy as np

def preprocess(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold of orange in HSV space
    lower_orange = np.array([0, 120, 200])
    upper_orange = np.array([25, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    color_masked = cv2.bitwise_and(image, image, mask = mask)

    _, gray, _ = cv2.split(color_masked)

    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=2)

    return gray