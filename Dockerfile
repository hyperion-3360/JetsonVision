FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

COPY entrypoint.sh ./
COPY ./src ./
COPY ./frameworks ./

RUN sudo apt-get update
RUN sudo apt-get install python3-pil.imagetk -y
RUN sudo apt-get install python3-tk -y
RUN sudo apt-get install ffmpeg -y
RUN sudo apt-get install vim -y

RUN python3 -m pip install -r requirements.txt

# Install the Teledyne Dalsa lib
RUN cp GigE-V-Framework_aarch64_2.21.1.0195.tar.gz $HOME
RUN cd $HOME
RUN tar -zxf GigE-V-Framework_aarch64_2.21.1.0195.tar.gz

RUN cd DALSA
RUN ./corinstall

# ENTRYPOINT ["./entrypoint.sh"]

# CMD ["python3", "autoencoder/training_context.py"]
