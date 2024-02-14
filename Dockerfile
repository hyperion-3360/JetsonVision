FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

COPY ./src ./
COPY ./pipdeps ./pipdeps

RUN python3 -m pip install --no-index --find-links /app/pipdeps/ -r requirements.txt
