FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN sudo apt-get update
RUN sudo apt-get -y install vim curl
RUN sudo apt-get install -y iputils-ping

COPY ./pipdeps ./pipdeps
COPY ./src/requirements.txt .
RUN python3 -m pip install --no-index --find-links /app/pipdeps/ -r requirements.txt

COPY ./src ./
