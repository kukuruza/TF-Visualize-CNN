from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import load

        
def generate_logs(model, logdir):
  ''' Generate logs for a model for graph visualization in Tensorboard. '''

  with tf.Session() as sess:
    load.load_model(model)
  train_writer = tf.summary.FileWriter(logdir)
  train_writer.add_graph(sess.graph)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True,
      help='''Could be either a directory containing the meta_file
           and ckpt_file or a model protobuf (.pb) file''')
  parser.add_argument('--logdir', type=str, required=True)
  args = parser.parse_args()
  
  generate_logs (args.model, args.logdir)
