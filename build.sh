docker build -t vision2024 .
docker volume create vision2024
docker run -it --rm --runtine nvidia --network host -v vision2024:/home/data vision2024

