FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04



RUN apt update -y && apt install -y python3.10 python3-pip virtualenv curl && apt autoremove
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN apt install -y git graphviz

COPY requirements.txt /root/
RUN pip3 install -r /root/requirements.txt

