# if you want to change the path
# you'll need to:
# export DETECTRON2_DATASETS=/home/jsimonelli/data/detectron2
# or 
# import os
# os.environ['DETECTRON2_DATASETS'] = '/home/jsimonelli/data/detectron2'

# detectron2 expects it to be in a folder called coco
mkdir coco
cd coco

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

unzip annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip