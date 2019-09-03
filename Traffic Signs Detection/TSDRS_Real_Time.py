print("[INFO] Importing libraries...")
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements 
from PIL import Image

import cv2
cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

print("[INFO] Defining global constants...")
# What model to download.
# F-RCNN INCEPTION V2
MODEL_NAME = 'TSDRS_FOUR_FRCNN_INCV2_inference_graph'
PATH_TO_LABELS = os.path.join('data', 'TSDRS_label_map_four.pbtxt')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


# Load a (frozen) Tensorflow model into memory
print("[INFO] Loading inference graph...")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Loading label map
print("[INFO] Loading label map...")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# configure Tensorflow to utilize the GPU and CUDA
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def tsdrs_rt_main():
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
      while True:
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
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

        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(27) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

if __name__ == '__main__':
  tsdrs_rt_main()