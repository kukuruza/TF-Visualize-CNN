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
import cv2  # For IO
from progressbar import ProgressBar

from recfield import ReceptiveField
from activations import Activations


parser = argparse.ArgumentParser()
parser.add_argument('--facenet_dir', type=str, required=True,
    help='Path to Facenet repo.')
parser.add_argument('--lfw_dir', type=str, required=True,
    help='Path to the data directory containing aligned LFW face patches.')
parser.add_argument('--model_path', type=str, required=True,
    help='A model protobuf (.pb) file.')
parser.add_argument('--num_channels', type=int, default=10,
    help='number of channels to visualize')
parser.add_argument('--num_samples', type=int, default=5,
    help='number of images to show for each feature.')
parser.add_argument('--layers', nargs='+',
    help='Names of layer tensors in the model.',
    default=[
          'InceptionResnetV1/Conv2d_2b_3x3/convolution:0',
          'InceptionResnetV1/Conv2d_4a_3x3/convolution:0',
          'InceptionResnetV1/Conv2d_4b_3x3/convolution:0',
          'InceptionResnetV1/Repeat/block35_1/add:0',
parser.add_argument('--out_dir', type=str, default='/tmp/facenet_activations',
    help='Output directory for visualizations.')
parser.add_argument('--logging_level', type=int, choices={10,20,30,40,50},
    default=30, help='Logging level 10 (verbose) to 50 (fatal).')
parser.add_argument('--print_all_names', action='store_true',
    help='Print the names of all tensors to choose from and exit.')
parser.add_argument('--batch_size', type=int, default=10,
    help='Batch size should be small enough to fit into your GPU.')
parser.add_argument('--batches', type=int, default=10000,
    help='Number of batches in the dataset to use. Training set contains 390 batches.')
FLAGS = parser.parse_args()

assert os.path.exists(FLAGS.facenet_dir), 'FaceNet does not exist at %s.' % FLAGS.facenet_dir
sys.path.insert(0, os.path.join(FLAGS.facenet_dir, 'src'))
import facenet
import lfw

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

  # Read the file containing the pairs used for testing
  pairs = lfw.read_pairs(os.path.join(FLAGS.facenet_dir, 'data/pairs.txt'))

  # Get the paths for the corresponding images
  lfw_dir = os.path.join(os.getenv('FACENET_DIR'), FLAGS.lfw_dir)
  paths, _ = lfw.get_paths(lfw_dir, pairs)
  # Remove duplicates,
  paths = list(set(paths))

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

  # Will store an Activation object for each layer for each channel.
  # The key is layer_name, the value is a list of 'Activations' objects, one per channel.
  activations = {}

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

    height, width, channels = IMAGE_SHAPE[1:]

    # Get all the features we want to visualize.
    layers_names = FLAGS.layers
    layers = [tf.get_default_graph().get_tensor_by_name(x) for x in layers_names]

    with tf.Session() as sess:
      
      # 1. Run through all images and find the best for each layer.
      #    'Activation' class is used in this loop.

      for i in ProgressBar()(range(min(len(paths) // FLAGS.batch_size, FLAGS.batches))):
        start_index = i * FLAGS.batch_size
        end_index = min((i + 1) * FLAGS.batch_size, FLAGS.batches * FLAGS.batch_size)
        paths_batch = paths[start_index:end_index]

        # Run a forward pass with normalized images.
        images = facenet.load_data(paths_batch, False, False, height)
        features = sess.run(layers, feed_dict={ images_ph: images, phase_train_ph: False })

        images_uint8 = facenet.load_data(
            paths_batch, False, False, height, do_prewhiten=False).astype(np.uint8)

        for ilayer, layer_name in enumerate(layers_names):
          # Lazy intialization.
          num_channels = min(FLAGS.num_channels, features[ilayer].shape[3])
          if layer_name not in activations:
            activations[layer_name] = [Activations(FLAGS.num_samples) for i in range(num_channels)]
          # Update activations.
          for ichannel in range(num_channels):
            activations[layer_name][ichannel].update(
                images_uint8, features=features[ilayer][:,:,:,ichannel])


      # 2. Run through layers and visualize images for the best features.
      #    'Recfield' class is used in this loop with 'Activation' class.

      for layer, layer_name in zip(layers, layers_names):

        # Compute receptive field for this layer and show statistics.
        forward = lambda x: sess.run([layer], feed_dict={ images_ph: x, phase_train_ph: False })
        recfield = ReceptiveField(IMAGE_SHAPE, forward)

        print (layer_name, 'feature map is of shape', layer.get_shape(),
               'and has receptive field', recfield.max())

        for ichannel in range(len(activations[layer_name])):
          image, crop = activations[layer_name][ichannel].retrieve(recfield)
          if ichannel == 0:
            images = image
            crops = crop
          else:
            images = np.concatenate((images, image), axis=1)
            crops = np.concatenate((crops, crop), axis=1)
        filename = layer_name.replace('/', '-').replace(':', '')
        cv2.imwrite(os.path.join(FLAGS.out_dir, '%s-full.jpg' % filename), images)#[:,:,::-1])
        cv2.imwrite(os.path.join(FLAGS.out_dir, '%s-crop.jpg' % filename), crops)#[:,:,::-1])

            
if __name__ == '__main__':
  main()
