from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import numpy as np
import cv2  # For visualization and io.
import tensorflow as tf
from progressbar import ProgressBar

from recfield import ReceptiveField
from activations import Activations


tf.app.flags.DEFINE_string('cifar10_dir', None,
    help='Path to the directory with TF tutorial on Cifar10.')
tf.app.flags.DEFINE_string('layers', default='conv1/conv1:0,conv2/conv2:0',
    help='Names of the model layers, separated by comma')
tf.app.flags.DEFINE_integer('num_channels', default=10,
    help='number of channels to visualize')
tf.app.flags.DEFINE_integer('num_samples', default=5,
    help='number of images to show for each feature.')
tf.app.flags.DEFINE_string('train_dir', default='/tmp/cifar10_train', 
    help='Directory with trained cifar10 model.')
tf.app.flags.DEFINE_string('out_dir', default='/tmp/cifar10_activations', 
    help='Directory for visualization results.')
tf.app.flags.DEFINE_integer('logging_level', default=30,
    help='Logging level 10 (verbose) to 50 (fatal).')
tf.app.flags.DEFINE_boolean('print_all_names', default=False,
    help='Print the names of all tensors to choose from and exit.')
tf.app.flags.DEFINE_integer('batches', default=390,
    help='Number of batches in the dataset to use. Training set contains 390 batches.')
FLAGS = tf.app.flags.FLAGS

assert os.path.exists(FLAGS.cifar10_dir), 'Cifar10 does not exist at %s.' % FLAGS.cifar10_dir
sys.path.insert(0, FLAGS.cifar10_dir)
import cifar10


def main():

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

  # Will store an Activation object for each layer for each channel.
  # The key is layer_name, the value is a list of 'Activations' objects, one per channel.
  activations = {}

  with tf.Graph().as_default():

    inputs, _ = cifar10.inputs(eval_data=False) # Training set.
    imshape = inputs.get_shape().as_list()
    print('Image shape:', imshape)

    # Logits takes images_ph, not inputs so that we can feed a generated image.
    images_ph = tf.placeholder(dtype=tf.float32, shape=imshape)
    logits = cifar10.inference(images_ph)

    # If you don't know which layers to use, print them out.
    if FLAGS.print_all_names:
      all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
      for name in all_names: print('%s:0' % name)
      return

    # # Get all the features we want to visualize.
    layers_names = FLAGS.layers.split(',')
    layers = [tf.get_default_graph().get_tensor_by_name(x) for x in layers_names]

    # Add an op to initialize the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
      assert latest_checkpoint is not None
      saver.restore(sess, latest_checkpoint)
      
      # 1. Run through all images and find the best for each layer.
      #    'Activation' class is used in this loop.

      for i in ProgressBar()(range(FLAGS.batches)):

        # Get a batch of images.
        images = sess.run(inputs)
        images_uint8 = np.stack([
          ((im - im.min()) / (im.max() - im.min()) * 255.).astype(np.uint8) for im in images])

        # Run a forward pass with normalized images.
        features = sess.run(layers, feed_dict={ images_ph: images })

        for ilayer, layer_name in enumerate(layers_names):
          # Lazy intialization.
          num_channels = min(FLAGS.num_channels, features[ilayer].shape[3])
          if layer_name not in activations:
            activations[layer_name] = [Activations(FLAGS.num_samples) for i in range(num_channels)]
          # Update activations for each channel.
          for ichannel in range(num_channels):
            activations[layer_name][ichannel].update(
                images_uint8, features=features[ilayer][:,:,:,ichannel])

      # 2. Run through layers and visualize images for the best features.
      #    'Recfield' class is used in this loop with 'Activation' class.

      for layer, layer_name in zip(layers, layers_names):
        
        forward = lambda x: sess.run([layer], feed_dict={ images_ph: x })

        # Compute receptive field for this layer and show statistics.
        recfield = ReceptiveField(inputs.get_shape().as_list(), forward)
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
        cv2.imwrite(os.path.join(FLAGS.out_dir, '%s-full.jpg' % filename), images[:,:,::-1])
        cv2.imwrite(os.path.join(FLAGS.out_dir, '%s-crop.jpg' % filename), crops[:,:,::-1])

      coord.request_stop()
      coord.join(threads)
       
if __name__ == '__main__':
  logging.basicConfig(level=FLAGS.logging_level)
  main()
