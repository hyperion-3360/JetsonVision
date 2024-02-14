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