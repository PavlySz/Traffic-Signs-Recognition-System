print("[INFO] Importing libraries...")
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO

from PIL import Image
import cv2

import pyttsx3

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util


print("[INFO] Defining global constants...")

# FRCNN_Incv2
MODEL_NAME = 'TSDRS_FOUR_FRCNN_INCV2_inference_graph'
PATH_TO_LABELS = os.path.join('data', 'TSDRS_label_map_four.pbtxt')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

engine = pyttsx3.init()

# ## Load a (frozen) Tensorflow model into memory
print("[INFO] Loading inference graph...")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
print("[INFO] Loading label map...")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 36) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# configure Tensorflow to utilize the GPU and CUDA
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def tsdrs_static_images_main():
  print("[INFO] Creating a session...")
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
      for image_path in TEST_IMAGE_PATHS:
          print("[INFO] Pre-processing image...")
          image = Image.open(image_path)
          image_np = load_image_into_numpy_array(image)

          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          
          # Actual detection
          print("[INFO] Detecting image...")
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          
        #   print(classes[0][0], scores[0][0])
          

          # Visualization of the results of a detection
          print("[INFO] Visualizing detected image...")
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              # min_score_thresh=0.2,
              line_thickness=8)


          print("[INFO] Showing detected image...")
          image_RGB = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          cv2.imshow("Image RGB", image_RGB)

          # ## NotificationSystem ## #
          predictions = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.5]
          if predictions:
            predicted_class = predictions[0]['name']
            print("I see a {} sign with confidence {:.1f}%".format(predicted_class, scores[0][0]*100))

            class_img = cv2.imread('{}.jpg'.format(predicted_class))
            cv2.imshow('{}'.format(predicted_class), class_img)

            engine.say(predicted_class)
            engine.runAndWait()

          cv2.waitKey(0)
          cv2.destroyAllWindows()

      print("DONE")
      print("HAIL HYDRA!")

if __name__ == '__main__':
  tsdrs_static_images_main()