FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

RUN sudo apt-get update
RUN sudo apt-get install
RUN sudo apt-get install vim curl -y

COPY ./pipdeps ./pipdeps
COPY ./src/requirements.txt .
RUN python3 -m pip install --no-index --find-links /app/pipdeps/ -r requirements.txt

COPY ./src ./
