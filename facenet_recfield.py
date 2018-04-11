from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import argparse
import requests
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2  # For IO.

from recfield import ReceptiveField


parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=True,
    help='A model protobuf (.pb) file.')
parser.add_argument('--out_dir', type=str, default='/tmp/facenet_recfield',
    help='Output directory for visualizations.')
parser.add_argument('--logging_level', type=int, choices={10,20,30,40,50},
    default=30, help='Logging level 10 (verbose) to 50 (fatal).')
parser.add_argument('--layers', nargs='+',
    help='Names of layer tensors in the model.',
    default=[
          'InceptionResnetV1/Conv2d_2b_3x3/convolution:0',
          'InceptionResnetV1/Conv2d_4a_3x3/convolution:0',
          'InceptionResnetV1/Conv2d_4b_3x3/convolution:0',
          'InceptionResnetV1/Repeat/block35_1/add:0',
parser.add_argument('--print_all_names', action='store_true',
    help='Print the names of all tensors to choose from and exit.')
parser.add_argument('--batch_size', type=int, default=128,
    help='Batch size should be small enough to fit into your GPU.')
FLAGS = parser.parse_args()

IMAGE_SHAPE = [FLAGS.batch_size, 160, 160, 3]


def load_model(model_path):
  ''' Load a model from a graph saved in .pb file. '''

  print('Model filename: %s' % model_path)
  assert os.path.exists(model_path)
  assert os.path.splitext(model_path)[1] == '.pb'

  with gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def main():

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

  with tf.Graph().as_default() as graph:

    # Load .pb model file.
    # It is a "freezed" model, where all variables are replaced with constants.
    load_model(FLAGS.model_path)

    # If you don't know which layers to use, print them out.
    if FLAGS.print_all_names:
      all_names = [n.name for n in graph.as_graph_def().node]
      for name in all_names: print('%s:0' % name)
      return

    images_ph = graph.get_tensor_by_name("input:0")
    phase_train_ph = graph.get_tensor_by_name("phase_train:0")
    
    with tf.Session() as sess:
      
      for layer_name in FLAGS.layers:
        
        # Make a forward function for this layer.
        layer = tf.get_default_graph().get_tensor_by_name(layer_name)
        forward = lambda x: sess.run([layer], feed_dict={ images_ph: x, phase_train_ph: False })

        # Compute receptive field for this layer and show statistics.
        recfield = ReceptiveField(IMAGE_SHAPE, forward)
        print ('Layer', layer_name, 'is of shape', layer.get_shape(),
               'and has receptive field', recfield.max())
        example = recfield.draw_example()
        filename = 'recfield-%s.png' % layer_name.replace('/', '-').replace(':0', '')
        cv2.imwrite(os.path.join(FLAGS.out_dir, filename), example)

            
if __name__ == '__main__':
  main()
