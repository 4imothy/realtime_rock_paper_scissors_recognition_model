# Real Time Rock-Paper-Scissors Detector AI With [Tensorflow Object Detection](https://www.tensorflow.org/hub/tutorials/object_detection)

Project created winter 2022-2023 to learn more about tensorflow.

### Demo


https://user-images.githubusercontent.com/40186632/215922232-3d0b0e16-5c90-43ca-b8ed-bba62ac0a3eb.mp4


## How to set up a real time object detection project on Mac. 

Create an issue if you have any questions.

#### My versions:
- Python: 3.10.6
- openCV: 4.5.5.64
- tensorflow: 2.11.0
- tensorflow\_hub: 0.12.0
- tensorflow\_io: 0.29.0
- protobuf: 21.12
- protoc: 3.21.12
- qt@5: 5.15.7

### Image Collection
1. Go to project directory and type
```
python3 -m venv tfod
```
to create an enviornment.

2. Start the enviornment, source tfod/bin/activate

3. Install openCV
```
pip install openCV-python
```

4. Open a python script to collect images, add this code
```
from cv2 import cv2
import uuid
import os
import time

# I am doing rock paper scissors
labels = ['rock', 'paper','scissors']
number_imgs = 6 # this is the number of images that will be collected for each class

# create images folder
IMAGES_PATH = os.path.join("images")

if not os.path.exists(IMAGES_PATH):
        os.mkdir(IMAGES_PATH)
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)

for label in labels:
    cap = cv2.VideoCapture(0) # this number might have to be changed depending on number of cameras in your workspace
    print('Collecting images for {}'.format(label))
    time.sleep(3)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        print('Got image {}'.format(imgnum))
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()
```

5. Run the script changing poses when the message printed to terminal says. This will collect the images and put them in the correct repository.

If you get an error that says something similar to "ImportError: Bindings generation error. Submodule name should always start with a parent module name. Parent name: cv2.cv2. Submodule name: cv2", run these commands to install a version that doesn't have this issue.
```
pip uninstall opencv-python
pip install opencv-python==4.5.5.64
```

6. Now that you have the images we will use [this](https://github.com/tzutalin/labelImg) github repo to label them
Install dependencies with
```
brew install qt@5
```
Make sure qmake is in your path. For me it was installed at */opt/homebrew/bin/qmake*
```
pip install lxml
pip install pyqt5 --config-settings --confirm-license= --verbose
```
This one will take a while.

7. Do this command to install the label image tool.
```
git clone https://github.com/tzutalin/labelImg
```
Go to the directory with the repository and run
```
make qt5py3
```
Then run the program with
```
python labelImg.py
```

8. Go through your images and label them with tight rectangles.
    1. Open dir, select the folder with class you want to label.
    2. Select an image press "w" to draw a rectangle around the object you want to be detected.
    3. Label the selected area, be sure to remember these labels you will use them again later.
    4. "Cmd+S" to save. Do this for every image in every image folder.
After that go to each image folder and make sure that there is a *.xml* file for each image.

9. Create two folders *train* and *test* in your images folder. Move most of the images and their *.xml* files to train and move some of the images and their associated *.xml* files to test.

### Training

1. Clone the tensorflow models repository
```
git clone --depth 1 https://github.com/tensorflow/models
```
2. Install the protobuf compiler
```
brew install protobuf-c
```

3. Compile the *.proto* files and setup with
```
cd models/research/
protoc object_detection/protos/*.proto --python\_out=.
cp object\_detection/packages/tf2/setup.py . ; python -m pip install .
```
When doing the last command you might get errors which is fine.

4. Run
```
python object_detection/builders/model_builder_tf2_test.py
```
Install the packages that it says are missing. I had to do
```
pip install absl-py tensorflow-macos tensorflow-metal tf_slim scipy matplotlib
pip install -U tf-models-official
```
If you get error about *tensorflow\_io* then go to your virtual enviornment folder (tfod) and follow these steps:
```
git clone https://github.com/tensorflow/io.git
cd io
python setup.py -q bdist_wheel
```
Then run pip install with that *.whl* file in the *dist* directory
```
pip install --no-deps dist/tensorflow_io-0.29.0-cp310-cp310-macosx_10_9_universal2.whl
```
Continue running until it works or until you get an error "cannot import name 'builder' from 'google.protobuf.internal'". At that point go to the project directory (the one containing the tfod folder) and run this line:
```
curl https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py > "./tfod/lib/python3.10/site-packages/google/protobuf/internal/builder.py"
```
Or copy the same file that is on this repository and place it in the position described in the command above.
This downloads the builder and places it in the correct location. Then run the test script again.

If you get an error about "No module named 'official'" follow these steps after going to the project directory then add the path to the models to your python path. You can add this line to *tfod/bin/activate* so it gets added every time you start your virtual enviornment.
```
export PYTHONPATH=$PYTHONPATH:/path_to_project/models
pip3 install -r models/official/requirements.txt
```
If you get errors relating to this in the future be sure to check your python path with
```
echo $PYTHONPATH
```
before further debugging.

After this step the scipt ran with "OK (skipped=1)" as the output meaning the object detection library was successfully installed.

We will use this model: ssd\_mobilenet\_v2\_fpnlite\_320x320\_coco17\_tpu-8, but you can see others (here)[https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md]

5. Go to project directory and create a folder for the pre-trained model with mkdir. Then install the model with curl.
```
mkdir pretrained_model
curl http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz > "./pretrained_model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
tar -zxvf pretrained_model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
rm -r pretrained_model
mv pretrained_model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz models
```
This will create a folder named *ssd\_mobilenet\_v2\_fpnlite\_320x320\_coco17\_tpu-8* in the models folder.

6. Create a python script to create a label map.
```
import os

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

labels = [{'name':'ThumbsUp', 'id':1}, {'name':'ThumbsDown', 'id':2}, {'name':'ThankYou', 'id':3}, {'name':'LiveLong', 'id':4}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
```

Create the directory *annotations* for the map to be stored in.
Then run the code.

7. Clone this repo to create tf records:

```
git clone https://github.com/nicknochnack/GenerateTFRecord
```

Run these lines to make the images tf records.

```
python GenerateTFRecord/generate_tfrecord.py -x images/train -l annotations/label_map.pbtxt -o annotations/train.record
python GenerateTFRecord/generate_tfrecord.py -x images/test -l annotations/label_map.pbtxt -o annotations/test.record
```

8. Create a folder for your learner and copy the *pipeline.config* file to it.

```
mkdir learner
cp models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config learner
```

9. Create a new file that will configure the learner for our project. Add these lines:

```
import tensorflow as tf
import os
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file("learner/pipeline.config")

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(os.path.join("learner","pipeline.config"), "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 3 # the number of classes
pipeline_config.train_config.batch_size = 5 # the number of images for each class
pipeline_config.train_config.fine_tune_checkpoint = os.path.join("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8","checkpoint","ckpt-0")
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= os.path.join("annotations","label_map.pbtxt")
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join("annotations","train.record")]
pipeline_config.eval_input_reader[0].label_map_path = os.path.join("annotations","label_map.pbtxt")
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join("annotations","test.record")]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile("learner/pipeline.config", "wb") as f:
    f.write(config_text)
```
Run the file to configure. Might be warnings.

10. Train the model by running this command.

```
python models/research/object_detection/model_main_tf2.py --model_dir=learner --pipeline_config_path=learner/pipeline.config --num_train_steps=3000
```

If you get errors about missing modules just install them.

```
pip install lvis
```

11. Evaluate the model with this command

```
python models/research/object_detection/model_main_tf2.py --model_dir=learner --pipeline_config_path=learner/pipeline.config --checkpoint_dir=learner
```
To get more information use tensor board:
```
tensorboard --logdir=.
```

12. Detecting with image. Load the model with
```
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
```
Add these lines to detect on the image given:
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

category_index = label_map_util.create_category_index_from_labelmap(os.path.join("annotations","label_map.pbtxt"))
IMAGE_PATH = os.path.join("collected_images","test","paper.573813d4-8d11-11ed-b9d2-16cb2971498c.jpg")

img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()
```

13. Add this text to a new file to detect in real time.
```
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join("learner","pipeline.config"))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("learner", 'ckpt-4')).expect_partial()

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(os.path.join("annotations","label_map.pbtxt"))

# detection_classes should be ints.

label_id_offset = 1


cap = cv2.VideoCapture()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
```
There might be errors in the editor, ignore.

I recommend going to *models/research/object_detection/utils/visualization_utils.py* and change the font size to be larger. On mac I had to copy from */Library/Fonts/Arial Unicode.ttf* to *./arial.ttf* and add these lines.
```
  try:
    font = ImageFont.truetype('./arial.ttf', 25)
  except IOError:
    font = ImageFont.load_default()
```
Or change *line_thickness*
```
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 10,
      'keypoint_edges': keypoint_edges
  }
```
