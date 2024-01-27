FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

COPY entrypoint.sh ./
COPY ./src ./
COPY ./frameworks ./

RUN sudo apt-get update
RUN sudo apt-get install python3-pil.imagetk -y
RUN sudo apt-get install python3-tk -y

# RUN python3 -m pip install -r requirements.txt

# ENTRYPOINT ["./entrypoint.sh"]

# CMD ["python3", "autoencoder/training_context.py"]
