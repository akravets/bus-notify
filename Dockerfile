FROM python:3.6.8

WORKDIR /camera
COPY . /camera

#RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install opencv-python
CMD [ "python", "camera.py" ]
