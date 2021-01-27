# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM tensorflow/tensorflow:2.4.0-gpu

# Install linux packages
RUN apt update && apt-get -y install git && apt install -y libgl1-mesa-glx

# Install python dependencies
RUN pip install --upgrade pip
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# Create working directory
RUN mkdir -p /opt/nuclio
WORKDIR /opt/nuclio
COPY . /opt/nuclio
RUN pip install -r requirements.txt
# Copy contents
COPY /home/ub/model/car288_0121_last.pb /opt/nuclio/model.pb