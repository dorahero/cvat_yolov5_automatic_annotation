# licencse plate detect
- [x] yolov5
- [x] cvat
- [x] tensorflow
- [x] automatic-annotation

# Usage cvat:
[cvat](https://github.com/openvinotoolkit/cvat)
[automatic-annotation](https://github.com/openvinotoolkit/cvat/tree/develop/serverless/tensorflow/faster_rcnn_inception_v2_coco/nuclio)
## How to use
```shell
python detect_lp_num_new.py --source $DATA_PATH
```
### compare with using preprocessing or not
```shell
python compare.py --source $DATA_PATH --view-img
```
