FROM python:3.6.8
FROM ubuntu:18.10

WORKDIR /camera

#RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get update
RUN apt-get install -y zip unzip python3.6 python3-pip wget
RUN apt-get install -y libsm6 libxext6 libxrender1
RUN apt-get update

RUN python3.6 -m pip install pip --upgrade
RUN pip install opencv-contrib-python
RUN pip install tensorflow

RUN wget https://github.com/tensorflow/models/archive/master.zip -P /camera
RUN unzip /camera/master.zip

COPY . /camera

CMD ["python3", "camera.py"]
