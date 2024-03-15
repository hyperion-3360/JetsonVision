## Setup offline jetson with multiple cameras
ip address of jetson nano on robot: 10.33.60.68 (might be different)

1. Clone and pull the JetsonVision `git clone git@github.com:hyperion-3360/JetsonVision.git`
2. Navigate into the git: `cd JetsonVision`
3. Copy files to jetson via scp: `scp -r . aaeon@10.33.60.68:~/Documents/FRC2024/JetsonVision/`
4. ssh into the jetson `ssh aaeon@10.33.60.68`
5. Build the vision image with docker compose `docker-compose build vision-srv`
6. Launch the containers `docker-compose up -d`
7. The april tags and ai data should appear in smartdashboard (network tables tab)

**Possible issues:**
- If april tags are not easily recognized, the calibration file might be incorrect. Edit the command in docker-compose.yml and relaunch the containers with `docker-compose down` and the `docker-compose up -d`. Maybe a docker compose restart is possible. Needs to be investigated
- Multiple camera usage is possible. Line `16` in `docker-compose.yml` can be edited to add more devices and more camera calibration files. The first device is used for AI inference. All cameras are used for April tags recognition

```
16: command: sh -c "python3 robot_vision.py /dev/front --apriltag --roboflow -e env.json -c camera2.json --width 1280 --height 720"

# Can be changed for
python3 robot_vision.py /dev/front /dev/back --roboflow -e env.json -c camera2.json camera4.json --width 1280 --height 720

# notice the added device and added camera calibration file...
```

## USB camera order and the joys of using Nvidia Jetson

Shall you ever want to be able to reliably assign cameras with a given jetson USB port, you should definitely read this section. The default implementation of video4linux doesn't care about the USB port assignation. It means using cv2.VideoCapture(0) will get you the first detected camera by v4l but there is no guarantee it will be the same camera every boot...!!! The only way to garantee that is (no not using /dev/videoxxx as this too can change every boot) but rather use you good friend (sarcasm...) udev.

Copy the file 99-cameras.rules from this repository to the /etc/udev/rules.d/ and restart udev using :
sudo udevadm control --reload-rules && udevadm trigger

to get the specifics of each of your camera port use the following command:

udevadm info --name=/dev/video0 --attribute-walk

the important field is the KERNEL one

the rule file included in this repos is for 2 Microsoft life HD3000 mapping them to /dev/front and /dev/back

## Docker compose

### Build the docker images
`docker-compose build`

### Start and (re)build the docker containers
`-d` sends the process in the background and automatically restarts the containers when the device boots up

`docker-compose up --force-recreate -d`

### Stop the container

Stop without removing the containers:

`docker-compose stop`

### Stop and remove containers
Next `docker-compose up` will automatically re-create the containers:

`docker-compose down`

### P.S.
Edit the [docker-compose.yml](docker-compose.yml) file to remove usb devices if some cameras are not plugged in or the camera ids changed

## Useful docker commands

**To manually build and launch docker containers, go to the [Steps to build and execute a docker image in a Jetson for AI and image processing](#steps-to-build-and-execute-a-docker-image-in-a-jetson-for-ai-and-image-processing) section**

### List running containers :
`docker container ls`

### Launch shell inside of container :
1. Copy the container id found with `docker container ls`
2. `docker exec -it [container-id] bash`

### Print container logs :
`docker container logs [container-id] --follow`

<br/>

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
