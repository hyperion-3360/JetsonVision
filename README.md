## April tags coordinates and rotation matrix

From the `Layout marking diagram` pdf:

**DIMENSIONS ARE IN INCHES**

> The XYZ Origin is established in the bottom left corner of the field (as viewed in the image above). An x coordinate of 0 is aligned with the Blue Alliance Station diamond plate. A y coordinate of 0 is aligned with the side border polycarbonate on the Scoring Table side of the field. A z coordinate of 0 is on the carpet.
+Z is up into the air from the carpet, +X is horizontal to the
right (in this image above) toward the opposing alliance stations, and +Y runs from the Field Border towards the SPEAKERS. The face-pose of the tags is denoted with 1 degree representation, the Z-rotation. 0° faces the red alliance
station, 90° faces the non- scoring table side, and 180°
faces the blue alliance station. Distances are measured to the center of the tag.

[**Calculate the rotation matrix around the Z axis**](https://www.redcrab-software.com/en/Calculator/3x3/Matrix/Rotation-Z)


### Convert rotation matrices in the FRC space to camera space

**FRC space to camera space matrix**
<pre>
S = | 0 0 -1 |
    | 1 0  0 |
    | 0 -1 0 |
</pre>

To get the rotation matrix of an april tag in the camera space, simply multiply the rotation to the space transformation matrix:
<pre>
R<sub>c</sub> = R<sub>FRC</sub> * S
</pre>

See [tags2024-formatter](src/april_tags/tags2024-formatter.py) for the implementation details

## Direct UDP stream from the Jetson to a remote computer on the same local network

### Server (jetson)
**Set the correct input device (`/dev/video*`) and change the ip address for the target device's ip**

`gst-launch-1.0 v4l2src device=/dev/video0 ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay config-interval=10 pt=96 ! udpsink host=10.33.60.212 port=5000`

### Client (mac)
**Listen for UDP packets on the same port (5000 in this case)**

`gst-launch-1.0 -v udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse  ! avdec_h264 ! videoconvert ! autovideosink`

## Steps to build and execute a docker image in a Jetson for AI and image processing

### Build the image
`docker build -t [image tag] .`

### Create the volume
`docker volume create [volume name]`

### Run the container in an interactive shell with CUDA enabled
`docker run -it --rm --runtine nvidia --network host -v [volume name]:[where to mount in container] [image tag]`

## Example with a simple autoencoder neural network

```
cd example
docker volume create visionvolume

# Image and container to train the neural network
# The model is saved under `results` in the visionvolume volume
docker build -t vision .
docker run -it --rm --runtime nvidia --network host -v visionvolume:/home/data vision

# Inference request to the autoencoder. Returns floats describing the resulting image
docker build -t infervision -f "./infervision/Dockerfile" .
docker run -it --rm --runtime nvidia --network host -v visionvolume:/home/data infervision
```

## Run with devices (example with /dev/video0 and /dev/video1)
`docker run -it --rm --runtime nvidia --device /dev/video0 --device /dev/video1 --network host -v visionvolume:/home/data jetsonvision`


## Record clips with `record_clips.py` on the robot's Jetson
If running in a Docker container, make sure the storage location in the script is pointed to the mounted Docker volume

1. ssh into the Jetson `ssh aaeon@10.33.60.68`
2. `cd Documents/FRC2024/vision2024/`
3. Check your available devices: `ls /dev/video*`
4. Change the camera ids in `src/record_clips.py` for your available devices
5. Change the save location for the videos in the script
6. **Optional - Docker container**
    1. Build the image (the repo should have everything needed to build offline)
    <br />
    `docker build -t jetsonvision .`

    2. Launch the container with the available devices
    <br />
    `docker run -it --rm --runtime nvidia --device /dev/video0 --device /dev/video1 --network host -v visionvolume:/home/data jetsonvision`
7. Launch the recording script
  <br />
  `python3 ./src/record_clips.py` (or on Docker: `python3 record_clips.py`)
8. Stop the recording with `ctrl-c`
9. Transfer the recordings from the Jetson to your device with scp (change the video location as needed)
  <br />
  `scp -r aaeon@10.38.60.68:~/Documents/FRC2024/vision2024/test/data/clips/ .`
