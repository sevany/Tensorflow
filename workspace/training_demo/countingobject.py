
from itertools import count
import json
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import cv2
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import collections
import csv
import time
json

from jsonya import id2name
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#from keras.preprocessing.image import save_img

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)




# What model
directPath = os.getcwd()
MODEL_NAME = '/home/sevtech/Documents/native_tensorflow_v2.6/start/Tensorflow/workspace/training_demo/models/cv_fasterrcnn'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(directPath, '/home/sevtech/Documents/native_tensorflow_v2.6/start/Tensorflow/workspace/training_demo/models/cv_fasterrcnn/mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90
THRESHOLD = 0.5

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
videofile = os.path.join(directPath, '/home/sevtech/Documents/native_tensorflow_v2.6/start/Tensorflow/workspace/training_demo/testing_video_highres/360/videotiga360p.mp4') 
# videofile_02 = os.path.join(directPath, '/home/sevtech/Downloads/testing_video_highres/videodua.mp4') 

VIDEO_SIZE = (640,360)

MIN_CONF_THRESH = 0.5
# VIDEO_SIZE_2 = (1200,900)

cap = cv2.VideoCapture(videofile)  
# cap_2 = cv2.VideoCapture(videofile_02)

# if (cap.isOpened()== False): 
#   print("Error opening video stream or file")
# if (cap_2.isOpened()== False): 
#   print("Error opening video stream or file")


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('outputtiga.avi',fourcc, 20, VIDEO_SIZE)

# fourcc_dua = cv2.VideoWriter_fourcc(*'MJPG')
# out_dua = cv2.VideoWriter('outputdua.avi',fourcc_dua, 20, VIDEO_SIZE_2 )

assert cap.isOpened(), 'Cannot capture source'
# assert cap_2.isOpened(), 'Cannot capture source'



frame_counter = 0
start = time.time()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            ret, image_np1 = cap.read()
            frame_counter +=1
            imH, imW, _ = image_np1.shape
            # ret_2, image_np2 = cap_2.read()
            image_np = cv2.resize(image_np1, VIDEO_SIZE)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # count = 0
            height, width, _ = image_np.shape
            final_score = np.squeeze(scores)    
            count = 0
            for i in range(100):
                if scores is None or final_score[i] > 0.5:
                        count = count + 1


            results = []
            for idx, class_id in enumerate(classes[0]):
                    # count += 1
                    conf = scores[0, idx]
                # for i in range(len(scores)):
                    # if conf[i] > MIN_CONF_THRESH:
                    if conf > THRESHOLD:
                        
                        bbox = boxes[0, idx]
                        ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
                            
                        results.append({"label": id2name[class_id],
                                        "counting": count,
                                        "prob": str(conf),
                                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
                        })




            output = {
                'label': 'Object Detection Result',
                'type': 'detection',
                'status': 200,
                'output': results
            }
            # print(output)

            # print(len(boxes.shape))


            with open('Outpunya.json', 'w') as file:
                json.dump(results, file)

            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

            path = 'testing_gpu/'+ '_output.csv'
            with open(path, 'wb') as write_file:
                writer = csv.writer(write_file)


             
            # Draw framerate, current count, and total count in corner of frame
            cv2.putText (image_np,'Total Detections : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)

            out.write(image_np)
            # out_dua.write(image_np)
            if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                        
            # # Display output
            # cv2.imshow('object detection', image_np_2)
                        # Display output
            cv2.imshow('object detection', image_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
out.release()
cv2.destroyAllWindows()


