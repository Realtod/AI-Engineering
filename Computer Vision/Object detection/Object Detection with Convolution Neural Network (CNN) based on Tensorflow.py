from IPython.display import display, Javascript, Image
import re
import zipfile
from base64 import b64decode
import sys
import os
import json

import io
import random

import tarfile
from datetime import datetime
from zipfile import ZipFile
import six.moves.urllib as urllib
import PIL.Image
from PIL import Image

%%capture
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util, config_util
from object_detection.utils.label_map_util import get_label_map_dict
from skillsnetwork import cvstudio

%%capture
import os
if os.path.exists("content-latest.zip"):
    pass
else:
    !wget https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/content/data/content-latest.zip
    !wget https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/content/data/tfrecord.py
    
with zipfile.ZipFile('content-latest.zip', 'r') as zip_ref:
    zip_ref.extractall('')

# Initialize the CV Studio Client
cvstudioClient = cvstudio.CVStudio()

# Download All Images
cvstudioClient.downloadAll()

# Get the annotations from CV Studio
annotations = cvstudioClient.get_annotations()
labels = annotations['labels']

CHECKPOINT_PATH = os.path.join(os.getcwd(),'content/checkpoint')
OUTPUT_PATH = os.path.join(os.getcwd(),'content/output')
EXPORTED_PATH = os.path.join(os.getcwd(),'content/exported')
DATA_PATH = os.path.join(os.getcwd(),'content/data')
CONFIG_PATH = os.path.join(os.getcwd(),'content/config')
LABEL_MAP_PATH = os.path.join(DATA_PATH, 'label_map.pbtxt')
TRAIN_RECORD_PATH = os.path.join(DATA_PATH, 'train.record')
VAL_RECORD_PATH = os.path.join(DATA_PATH, 'val.record')

os.makedirs(DATA_PATH, exist_ok=True)
with open(LABEL_MAP_PATH, 'w') as f:
    for idx, label in enumerate(labels):
        f.write('item {\n')
        f.write("\tname: '{}'\n".format(label))
        f.write('\tid: {}\n'.format(idx + 1))
        f.write('}\n')

import tensorflow as tf
import os 
import io
import PIL.Image
from object_detection.utils import dataset_util, label_map_util, config_util
from object_detection.utils.label_map_util import get_label_map_dict

def create_tf_record(images, annotations, label_map, image_path, output):
  # Create a train.record TFRecord file.
  with tf.python_io.TFRecordWriter(output) as writer:
    # Loop through all the training examples.
    for image_name in images:
      try:
        # Make sure the image is actually a file
        img_path = os.path.join(image_path, image_name)   
        if not os.path.isfile(img_path):
          continue
          
        # Read in the image.
        with tf.gfile.GFile(img_path, 'rb') as fid:
          encoded_jpg = fid.read()

        # Open the image with PIL so we can check that it's a jpeg and get the image
        # dimensions.
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
          raise ValueError('Image format not JPEG')

        width, height = image.size

        # Initialize all the arrays.
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for a in annotations[image_name]:
          if ("x" in a and "x2" in a and "y" in a and "y2" in a):
            label = a['label']
            xmins.append(a["x"])
            xmaxs.append(a["x2"])
            ymins.append(a["y"])
            ymaxs.append(a["y2"])
            classes_text.append(label.encode("utf8"))
            classes.append(label_map[label])
       
        # Create the TFExample.
        tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(image_name.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        if tf_example:
          # Write the TFExample to the TFRecord.
          writer.write(tf_example.SerializeToString())
      except ValueError:
        print('Invalid example, ignoring.')
        pass
      except IOError:
        print("Can't read example, ignoring.")
        pass
    
    
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as PImage
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
DATA_PATH       = os.path.join(os.getcwd(),'content/data')
LABEL_MAP_PATH    = os.path.join(DATA_PATH, 'label_map.pbtxt')
EXPORTED_PATH   = os.path.join(os.getcwd(),'content/exported')
def displaydetectedobject(image):
# Load the labels
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# Load the model
    path_to_frozen_graph = os.path.join(EXPORTED_PATH, 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #image = PImage.open('nayef.jpeg')
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=(12, 8))
        return plt.imshow(image_np),print('Wears'+' '+  category_index[np.squeeze(classes).astype(np.int32)[0]]['name']+ ' '),scores[0][1]
# from tfrecord import create_tf_record, displaydetectedobject
image_files = [image for image in annotations["annotations"].keys()]

label_map = label_map_util.get_label_map_dict(LABEL_MAP_PATH)

random.seed(42)
random.shuffle(image_files)
num_train = int(0.7 * len(image_files))
train_examples = image_files[:num_train]
val_examples = image_files[num_train:]

create_tf_record(train_examples, annotations["annotations"], label_map, os.path.join(os.getcwd(),'images'), TRAIN_RECORD_PATH)
create_tf_record(val_examples, annotations["annotations"], label_map, os.path.join(os.getcwd(),'images'), VAL_RECORD_PATH)

MODEL_TYPE = 'ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18'
CONFIG_TYPE = 'ssd_mobilenet_v1_quantized_300x300_coco14_sync'
download_base = 'http://download.tensorflow.org/models/object_detection/'
model = MODEL_TYPE + '.tar.gz'
tmp = '/resources/checkpoint.tar.gz'

if not (os.path.exists(CHECKPOINT_PATH)):
    # Download the checkpoint
    opener = urllib.request.URLopener()
    opener.retrieve(download_base + model, tmp)

    # Extract all the `model.ckpt` files.
    with tarfile.open(tmp) as tar:
        for member in tar.getmembers():
            member.name = os.path.basename(member.name)
            if 'model.ckpt' in member.name:
                tar.extract(member, path=CHECKPOINT_PATH)
            if 'pipeline.config' in member.name:
                tar.extract(member, path=CONFIG_PATH)

    os.remove(tmp)

pipeline_skeleton = 'content/models/research/object_detection/samples/configs/' + CONFIG_TYPE + '.config'
configs = config_util.get_configs_from_pipeline_file(pipeline_skeleton)

label_map = label_map_util.get_label_map_dict(LABEL_MAP_PATH)
num_classes = len(label_map.keys())
meta_arch = configs["model"].WhichOneof("model")

override_dict = {
  'model.{}.num_classes'.format(meta_arch): num_classes,
  'train_config.batch_size': 6,
  'train_input_path': TRAIN_RECORD_PATH,
  'eval_input_path': VAL_RECORD_PATH,
  'train_config.fine_tune_checkpoint': os.path.join(CHECKPOINT_PATH, 'model.ckpt'),
  'label_map_path': LABEL_MAP_PATH
}

configs = config_util.merge_external_params_with_configs(configs, kwargs_dict=override_dict)
pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_config, DATA_PATH)

paths = [
    f'home/jupyterlab/conda/envs/python/lib/python3.6',
    f'content/models/research',
    f'content/models/research/slim'
]
os.environ['PYTHONPATH'] = ':'.join(paths)

%%capture
epochs = 40
start_datetime = datetime.now()
!python -m object_detection.model_main \
    --pipeline_config_path=$DATA_PATH/pipeline.config \
    --num_train_steps=$epochs \
    --num_eval_steps=100

regex = re.compile(r"model\.ckpt-([0-9]+)\.index")
numbers = [int(regex.search(f).group(1)) for f in os.listdir(OUTPUT_PATH) if regex.search(f)]
TRAINED_CHECKPOINT_PREFIX = os.path.join(OUTPUT_PATH, 'model.ckpt-{}'.format(max(numbers)))

!python3 -m object_detection.export_inference_graph \
  --pipeline_config_path=$DATA_PATH/pipeline.config \
  --trained_checkpoint_prefix=$TRAINED_CHECKPOINT_PREFIX \
  --output_directory=$EXPORTED_PATH
end_datetime = datetime.now()

from PIL import Image

# Here you can specify your own image 
URL = 'https://cdn.cliqueinc.com/posts/289533/kamala-harris-face-mask-289533-1602269219518-square.700x0c.jpg' 

with urllib.request.urlopen(URL) as url:
    with open('test.jpg', 'wb') as f:
        f.write(url.read())
image = Image.open('test.jpg')

%matplotlib inline
import numpy as np
n,img,accuracy=displaydetectedobject(image)

parameters = {
    'epochs': epochs
}

result = cvstudioClient.report(started=start_datetime, completed=end_datetime, parameters=parameters, accuracy= round(float(accuracy),2)*100)

if result.ok:
    print('Congratulations your results have been reported back to CV Studio!')

%%capture
!tensorflowjs_converter \
  --input_format=tf_frozen_model \
  --output_format=tfjs_graph_model \
  --output_node_names='Postprocessor/ExpandDims_1,Postprocessor/Slice' \
  --quantization_bytes=1 \
  --skip_op_check \
  $EXPORTED_PATH/frozen_inference_graph.pb \
  .
import json

from object_detection.utils.label_map_util import get_label_map_dict

label_map = get_label_map_dict(LABEL_MAP_PATH)
label_array = [k for k in sorted(label_map, key=label_map.get)]

with open(os.path.join('', 'labels.json'), 'w') as f:
    json.dump(label_array, f)
#!cd model_web 
with ZipFile('model_web.zip','w') as zip:
    zip.write('group1-shard1of2.bin')
    zip.write('group1-shard2of2.bin')
    zip.write('model.json')
    zip.write('labels.json')
    
# Download All Images
cvstudioClient.uploadModel('model_web.zip', {'epochs': epochs })

