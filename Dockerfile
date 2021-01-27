FROM tensorflow/tensorflow:2.4.0-gpu

RUN apt update && apt-get -y install git && apt install -y libgl1-mesa-glx

RUN pip install --upgrade pip

RUN mkdir -p /opt/nuclio
WORKDIR /opt/nuclio
COPY . /opt/nuclio

RUN pip install -r requirements.txt
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html