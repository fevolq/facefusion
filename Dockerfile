FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install ffmpeg -y

COPY . /facefusion
WORKDIR /facefusion
RUN cd /facefusion

RUN python install.py --onnxruntime cuda-11.8 --skip-conda

EXPOSE 7860
EXPOSE 8000

ENV CLI_ARGS=""

CMD python run.py ${CLI_ARGS}
