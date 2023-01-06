import os
import object_detection

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

labels = [{'name':'rock', 'id':1}, {'name':'paper', 'id':2}, {'name':'scissors', 'id':3}]

files = {
    'PIPELINE_CONFIG':os.path.join(CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join('annotations', LABEL_MAP_NAME)
}

# this will create a label map
with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
