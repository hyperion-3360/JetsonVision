## Static IP addresses of teledyne cameras
Camera 2: 169.254.2.146

## Steps to build and execute a docker image in a Jetson for AI and image processing

### Build the image
`docker build -t [image tag] .`

### Create the volume
`docker volume create [volume name]`

### Run the container in an interactive shell with CUDA enabled
`docker run -it --rm --runtine nvidia --network host -v [volume name]:[where to mount in container] [image tag]`

### To add camera access to Docker:
1. `ls /dev/video*`

2. In the docker command, add `--device /dev/video[0-9]` depending on your device's index

e.g.

`docker run -it --rm --runtine nvidia --network host --device /dev/video0 -v [volume name]:[where to mount in container] [image tag]`

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
