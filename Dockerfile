FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR app

COPY ./src ./

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["entrypoint.sh"]