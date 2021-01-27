FROM tensorflow/tensorflow:2.1.1-gpu

RUN apt update && apt-get -y install git && apt install -y libgl1-mesa-glx

RUN pip install --upgrade pip

RUN mkdir -p /opt/nuclio
WORKDIR /opt/nuclio

COPY . /opt/nuclio
RUN pip install -r requirements.txt

COPY /home/ub/model/car288_0121_last.pb /opt/nuclio/model.pb