# licencse plate detect
- [x] yolov5
- [x] cvat
- [x] tensorflow
- [x] automatic-annotation

# Usage cvat:
[cvat](https://github.com/openvinotoolkit/cvat)  
[automatic-annotation](https://github.com/openvinotoolkit/cvat/tree/develop/serverless/tensorflow/faster_rcnn_inception_v2_coco/nuclio)
## Build all cvat server
```shell
git clone https://github.com/openvinotoolkit/cvat.git
cd cvat
docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f components/analytics/docker-compose.analytics.yml up -d
```
## automatic-annotation
```shell
nuctl create project cvat
nuctl deploy tf-yolov5-gpu   --project-name cvat --path "serverless/tensorflow/yolov5/nuclio" --platform local   --base-image  dorahero2727/yolov5-tf-cvat:v1   --desc "YOLOv5 tensorflow" -i dorahero2727/yolov5-tf-cvat:v2 --resource-limit nvidia.com/gpu=1
```
