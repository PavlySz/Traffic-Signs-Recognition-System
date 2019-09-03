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
import pyttsx3

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

print("[INFO] Defining global constants...")
# F-RCNN INCEPTION V2
MODEL_NAME = 'TSDRS_FOUR_FRCNN_INCV2_inference_graph'
PATH_TO_LABELS = os.path.join('data', 'TSDRS_label_map_four.pbtxt')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

engine = pyttsx3.init()

### For video ###
VIDEO_FILE_INPUT = 'TSDRS.mp4'
cap = cv2.VideoCapture(VIDEO_FILE_INPUT)

SCALING_FACTOR = 50

FRAME_WIDTH = int(cap.get(3))
FRAME_HEIGHT = int(cap.get(4))
OUT_FPS = 30

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), OUT_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
 

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    rescaled_frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
    return rescaled_frame


RESCALED_FRAME_WIDTH = int(FRAME_WIDTH * SCALING_FACTOR/100)
RESCALED_FRAME_HEIGHT = int(FRAME_HEIGHT * SCALING_FACTOR/100)
print("{}x{}".format(RESCALED_FRAME_WIDTH, RESCALED_FRAME_HEIGHT))

out_rescaled = cv2.VideoWriter('output_video_rescaled.avi', cv2.VideoWriter_fourcc('M','J','P','G'), OUT_FPS, (RESCALED_FRAME_WIDTH, RESCALED_FRAME_HEIGHT))
### End for video ###

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


# configure Tensorflow to utilize the GPU and CUDA
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("[INFO] Starting detection...")
def tsdrs_video_main():
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
      while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            scaled_frame = rescale_frame(frame, SCALING_FACTOR)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(scaled_frame, axis=0)
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
                scaled_frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # cv2.imshow('object detection', cv2.resize(scaled_frame, (800,600)))
            cv2.imshow('object detection', scaled_frame)

            # ## NotificationSystem ## #
            predictions = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.5]
            if predictions:
              predicted_class = predictions[0]['name']
              print("I see a {} sign with confidence {:.1f}%".format(predicted_class, scores[0][0]*100))

              class_img = cv2.imread('{}.jpg'.format(predicted_class))
              cv2.imshow('{}'.format(predicted_class), class_img)

              engine.say(predicted_class)
              engine.runAndWait()

            # out_rescaled.write(scaled_frame)

            if cv2.waitKey(27) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            print("Saved video output.avi")
            print("DONE!")

if __name__ == '__main__':
  tsdrs_video_main()