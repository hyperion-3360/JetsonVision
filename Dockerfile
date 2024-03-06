FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

COPY ./pipdeps ./pipdeps
COPY ./src/requirements.txt .
RUN python3 -m pip install --no-index --find-links /app/pipdeps/ -r requirements.txt

COPY ./src ./
