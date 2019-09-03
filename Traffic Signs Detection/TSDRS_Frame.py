### imports ###
print("[INFO] Importing libraries...")
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import socket
import cv2

import matplotlib
matplotlib.use('Agg') # pylint: disable=multiple-statements

from distutils.version import StrictVersion
from collections import defaultdict
from matplotlib import pyplot as plt

sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils as vis_util

from receive_frames import *
from process_frame import process_frame
### end imports ###

print("[INFO] Defining global constants...")
host = socket.gethostbyname('0.0.0.0')    # Reachable to all devices
port = 8000

# What model to load
# F-RCNN RESNET50
# MODEL_NAME = 'TSRP_FRCNN_RN50_inference_graph'
# PATH_TO_LABELS = os.path.join('data', 'TSRP_label_map.pbtxt')

# F-RCNN INCEPTION V2
MODEL_NAME = 'TSDRS_FOUR_FRCNN_INCV2_inference_graph'
PATH_TO_LABELS = os.path.join('data', 'TSDRS_label_map_four.pbtxt')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# Load inference graph
print("[INFO] Loading inference graph...")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Load label map
print("[INFO] Loading label map...")
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def func(sess, frame):
    print("[INFO] Processing frame...")
    image = process_frame(frame)

    image_np_expanded = np.expand_dims(image, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection
    print("[INFO] Detecting frame...")
    (boxes, scores, classes, nuam_detections)=sess.run(
	    [boxes, scores, classes, num_detections],
	    feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection
    print("[INFO] Visualizing detected frame...")
    vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        
    # show detected frame
    print("[INFO] Showing detected frame...")
    img = np.flip(np.rot90(image, 3), 1)    # Rotate 90 degrees and flip the image
    cv2.imshow('Detected image', img)
    cv2.waitKey(100)    # Show images for 0.1 seconds
    # cv2.destroyAllWindows()    # Commented in order to not close the window after every image
	

# Receive and detect frame
def tsdrs_frame_main():
    print("[INFO] Creating a session...")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            server_socket, conn = socket_init(host, port)

            while True:
                print("[INFO] Receiving frame...")
                frame = receive_one_frame(conn, 960)
                if not frame: break
			
                print("[INFO] Received frame length = {}".format(len(frame)))

                func(sess, frame)

                print("[INFO] Waiting to receive next frame. Close the client connection to terminate.")

            # close client connection and socket
            print("[INFO] Client disonnected!")
            conn.close()        
            server_socket.close()
            print("[INFO] Connection closed!")


if __name__ == '__main__':
    tsdrs_frame_main()
	
#print("HAIL HYDRA!")