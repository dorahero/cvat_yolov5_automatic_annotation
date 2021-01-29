# licencse plate detect
- [x] yolov5
- [x] cvat
- [x] tensorflow
- [x] automatic-annotation

# Usage cvat:
[cvat](https://github.com/openvinotoolkit/cvat)  
[automatic-annotation](https://github.com/openvinotoolkit/cvat/blob/develop/cvat/apps/documentation/installation_automatic_annotation.md)
## Build all cvat server
```shell
git clone https://github.com/openvinotoolkit/cvat.git
cd cvat
docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f components/analytics/docker-compose.analytics.yml up -d
```
## automatic-annotation
In the cvat floder.
```shell
mkdir serverless/tensorflow/yolov5/  
cd serverless/tensorflow/yolov5/  
git clone https://github.com/dorahero/cvat_yolov5_automatic_annotation.git
mv cvat_yolov5_automatic_annotation nuclio
```
Prepare to deploy function with nuctl.  
- put your yolov5-tensorflow model in nuclio floder
- and rename as model.pb
```shell
cd nuclio
cp ~/model/example.pb model.pb
```
- build your self docker image
```shell
docker build -t xxxxx/cvat-cu10:v1 .
```
- create nuctl project as cvat
- deploy function 
```shell
nuctl create project cvat
nuctl deploy tf-yolov5-gpu   --project-name cvat --path "serverless/tensorflow/yolov5/nuclio" --platform local   --base-image  xxxxx/cvat-cu10:v1   --desc "YOLOv5 tensorflow" -i xxxxx/cvat-cu10:v2 --resource-limit nvidia.com/gpu=1
```
