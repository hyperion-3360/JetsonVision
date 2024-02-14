import os
from pathlib import Path
import signal
from threading import Thread

import cv2

QUIT = False

def record(cameraid: int, store: Path):
  cap = cv2.VideoCapture(cameraid)
  if not cap.isOpened():
    print(f"Failed to init camera {cameraid}")
    return


  print(store / f"clip-{cameraid}.mp4")
  out = cv2.VideoWriter(str(store / f"clip-{cameraid}.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1280, 720))

  global QUIT
  while not QUIT: # meh we don't care about thread safety here. We're just reading and we assume only 1 thread will change the QUIT value. In the worst case, we will stop a couple of frames later
    r, f = cap.read()
    if(r):
      out.write(f)

  out.release()
  cap.release()

if __name__ == "__main__":
  def stop(sig, frame):
    global QUIT
    QUIT = True  # We should use a lock, but whatever.... It's probably atomic and the other threads using QUIT should only read its value

  signal.signal(signal.SIGINT, stop)

  # Get the storage location
  store = Path('./test/data/clip') # For Docker container, set this path to the mounted volume (/home/data)
  store.mkdir(parents=True, exist_ok=True)

  store = store / f"{1 + len(os.listdir(store))}" # folder is named by the take number. E.g. .....clip/0/ or .....clip/1/, etc.
  store.mkdir()

  # Start the recording
  cameras = [0, 2]  # camera ids

  threads = []
  for camera in cameras:
    t = Thread(target = record, args=[camera, store])
    threads.append(t)

    t.start()

  for t in threads:
    t.join()
