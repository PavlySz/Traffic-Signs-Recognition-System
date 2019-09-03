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
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util


print("[INFO] Defining global constants...")
# FRCNN_ResNet50 (1000)
# MODEL_NAME = 'TSRP_FRCNN_RN50_inference_graph'
# PATH_TO_LABELS = os.path.join('data', 'TSRP_label_map.pbtxt')

# FRCNN_Incv2
MODEL_NAME = 'TSDRS_FOUR_FRCNN_INCV2_inference_graph'
PATH_TO_LABELS = os.path.join('data', 'TSDRS_label_map_four.pbtxt')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


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

# configure Tensorflow to utilize the GPU and CUDA
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

times = []
def tsdrs_static_images_main():
    print("[INFO] Creating a session...")
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            for root, _, image_files in os.walk(PATH_TO_TEST_IMAGES_DIR):
                num_images = len(image_files)
                for image_file in image_files:
                    img_path = os.path.join(root, image_file)

                    print("[INFO] Pre-processing image...")
                    image = Image.open(img_path)
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
                    s_time = time.time()
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    
                    # Visualization of the results of a detection
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    
                    image_np_RGB = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                    e_time = time.time()
                    times.append(e_time - s_time)
                    if not os.path.isdir('TSDRS_results'):
                        os.mkdir('TSDRS_results')

                    out_path = os.path.join('TSDRS_results', image_file)
                    print("[INFO] Saving image {}".format(out_path))
                    cv2.imwrite("{}".format(out_path), image_np_RGB)
    sum = 0
    for t in times:
        sum += t
    avg_time = sum/len(times)
    print("Total time on {} images = {}".format(num_images, avg_time))
    print("Average time = {}".format(avg_time))

    print("DONE")
    print("HAIL HYDRA!")

if __name__ == '__main__':
  tsdrs_static_images_main()