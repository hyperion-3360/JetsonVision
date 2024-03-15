#######################################################################################
# Test udev mappings
# Some devices might show not as opened even though the mapping is correct.
# Seems like v4l2 doesn't like opening multiple cameras back to back
#######################################################################################

import cv2

front = cv2.VideoCapture("v4l2:///dev/front")
back = cv2.VideoCapture("v4l2:///dev/back")
driver = cv2.VideoCapture("v4l2:///dev/driver")

print(f"\n\n/dev/front is open ?\t {front.isOpened()}")
print(f"/dev/back is open ?\t {back.isOpened()}")
print(f"/dev/driver is open ?\t {driver.isOpened()}\n\n")

if front.isOpened(): front.release()
if back.isOpened(): back.release()
if driver.isOpened(): driver.release()
