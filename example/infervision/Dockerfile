FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

COPY ./vision ./

RUN python3 -m pip install -r requirements.txt

CMD ["python3", "autoencoder/infer_local.py"]
