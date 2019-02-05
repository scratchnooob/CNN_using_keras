import sys
import skvideo.io
import subprocess as sp


# coding: utf-8

# In[1]:


from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from flask import Flask, render_template, Response,request



from utils import label_map_util

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'pretrained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
def load_model():
	global model
	model = tf.keras.models.load_model(emotion_model_path, compile=False)
            # this is key : save the graph after loading the model
	global graph
	graph = tf.get_default_graph()
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
ef gen(myfile,height,width):
    #cap = cv2.VideoCapture(myfile,cv2.CAP_ANY)
    #cap = skvideo.io.vreader(myfile)
    pipe = sp.Popen([ 'ffmpeg', "-i", myfile,
           "-loglevel", "quiet", # no text output
           "-an",   # disable audio
           "-f", "image2pipe",
           "-pix_fmt", "bgr24",
           "-vcodec", "rawvideo", "-"],
           stdin = sp.PIPE, stdout = sp.PIPE)

    with detection_graph.as_default():
     with tf.Session(graph=detection_graph) as sess:
        while True:
          #print(cap.isOpened())
          #cap.open(myfile)
          #print(cap.isOpened())
          #ret, image_np = cap.read()i
          #image_np = next(cap)
          h = int(height)
          w = int(width)
          raw_image = pipe.stdout.read(h*w*3)
          image_np =  np.fromstring(raw_image, dtype='uint8').reshape((w,h,3))
          #print(cap)
          #print(ret)
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

          #cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
          ret, jpeg = cv2.imencode('.jpg', image_np)
          yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
          if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break







@app.route('/video_feed')
def video_feed():
    myfile = request.args.get("url")
    height = request.args.get("height")
    width = request.args.get("width")
    return Response(gen(myfile,height,width), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5000)

