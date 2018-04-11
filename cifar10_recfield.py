from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import numpy as np
import cv2  # For visualization and io.
import tensorflow as tf
from recfield import ReceptiveField


tf.app.flags.DEFINE_string('cifar10_dir', None,
    help='Path to the directory with TF tutorial on Cifar10.')
tf.app.flags.DEFINE_string('out_dir', default='/tmp/cifar10_recfield',
    help='Directory for visualization results.')
tf.app.flags.DEFINE_integer('logging_level', default=30,
    help='Logging level 10 (verbose) to 50 (fatal).')
tf.app.flags.DEFINE_string('layers', default='conv1/conv1:0,conv2/conv2:0',
    help='Names of the model layers, separated by comma')
tf.app.flags.DEFINE_boolean('print_all_names', default=False,
    help='Print the names of all tensors to choose from and exit.')
FLAGS = tf.app.flags.FLAGS

assert os.path.exists(FLAGS.cifar10_dir), 'Cifar10 does not exist at %s.' % FLAGS.cifar10_dir
sys.path.insert(0, FLAGS.cifar10_dir)
import cifar10
from cifar10_input import IMAGE_SIZE


def main():

  if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

  with tf.Graph().as_default():

    images_ph = tf.placeholder(dtype=tf.float32,
        shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    print('Image shape:', images_ph.get_shape())
    imshape = images_ph.get_shape().as_list()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images_ph)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # If you don't know which layers to use, print them out.
    if FLAGS.print_all_names:
      all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
      for name in all_names: print('%s:0' % name)
      return

    with tf.Session() as sess:
      
      sess.run([init_op])
      
      layer_names = FLAGS.layers.split(',')
      for layer_name in layer_names:
        
        # Make a forward function for this layer.
        layer = tf.get_default_graph().get_tensor_by_name(layer_name)
        forward = lambda x: sess.run([layer], feed_dict={ images_ph: x })

        # Compute receptive field for this layer and show statistics.
        recfield = ReceptiveField(imshape, forward)
        print (layer_name, 'feature map is of shape', layer.get_shape(),
               'and has receptive field', recfield.max())
        example = recfield.draw_example()
        filename = 'recfield-%s.png' % layer_name.replace('/', '-').replace(':0', '')
        cv2.imwrite(os.path.join(FLAGS.out_dir, filename), example)

            
if __name__ == '__main__':
  logging.basicConfig(level=FLAGS.logging_level)
  main()
