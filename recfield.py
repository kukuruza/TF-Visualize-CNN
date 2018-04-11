"""
Compute receptive field for each pixel of several selected layers of a CNN.

The file contains class ReceptiveField which does all the work.
Provide a function to do a forward pass for one layer to its constructor.
It will then compute the receptive fields for this layer.

WARNING: the forward pass function should NOT be in the "training mode",
otherwise batch_norm layers will kill the magic.

Internally the class generates and passes forward carefully designed images
and tracks which pixel in the image affects which pixel in the feature map.
That is, the receptive field is computed directly, by definition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np
import tensorflow as tf


class ReceptiveField():
  ''' Compute the receptive field for each pixel in several selected layers. '''

  @staticmethod
  def _build_batches_vertical(shape):
    ''' Prepare input batches for computing the vertical receptive field. '''

    assert len(shape) == 4, shape
    [batch_size, height, width, channels] = shape
    # The last image in a batch is completely black for reference.
    # Regardless of the type of images on the input, let's pass float.
    A = np.zeros([height + 1, height, width, channels], dtype=float)
    for i in range(height):
      A[i, i, :, :] = 1.
    # Pad batch dimension to cut an array into batches of the same shape.
    pad = (A.shape[0] // batch_size + 1) * batch_size - A.shape[0]
    A = np.pad(A, ((0,pad),(0,0),(0,0),(0,0)), 'constant')
    A = A[np.newaxis, ...].reshape([-1, batch_size, height, width, channels])
    return A
    
  @staticmethod
  def _interpret_vertical(maps):
    ''' Interpret a feature map produced from carefully designed input batches
    as vertical receptive field maps for this layer.
    '''
    #np.set_printoptions(threshold='nan', linewidth=200)
    assert len(maps.shape) == 4, maps.shape
    image_height, feature_height, feature_width, _ = maps.shape

    matrix = np.absolute(maps).sum(axis=-1)  # Sum finds a channel with nonzero output.
    offset = matrix[-1]  # The last image in the input batch was made black.
    offset = np.tile(offset, reps=(maps.shape[0],1,1))
    assert offset.shape == matrix.shape, offset.shape
    matrix = matrix != offset
    #print (matrix[:,:,0].astype(int))

    # For each feature in feature map find the first and the last visible pixel.
    recfield = np.zeros((feature_height, feature_width, 2), dtype=int)
    recfield[:,:,0] = matrix.argmax(axis=0)
    recfield[:,:,1] = image_height - matrix[::-1,:,:].argmax(axis=0)
    #print (recfield)
    np.testing.assert_array_less(recfield[:,:,0], recfield[:,:,1])
    return recfield

  @staticmethod
  def _run_forward(batches, forward_func):
    ''' Run forward-pass batch by batch, collect feature maps
    from individual batches into a single feature map.
    '''
    for ibatch, batch in enumerate(batches):
      out = forward_func(batch)
      assert len(out) == 1, 'forward_func must compute exactly one tensor.'
      if ibatch == 0:
        maps = out[0]
      else:
        maps = np.concatenate((maps, out[0]), axis=0)
        logging.debug('Added a batch, now of shape', maps.shape)
    assert len(maps.shape) == 4, maps.shape
    return maps

  def __init__(self, shape, forward_func):
    ''' Compute maps of receptive field for multiple layers of interest.

    Args:
      shape:         shape of a batch of input images, must be a list of len 4,
                     i.e. [batch_size, height, width, channels].
      forward_func:  function that takes a batch of images on input and returns
                     a tensor corresponding to the layer of interest.
                     It should be a wrapper around the TF sess.run() function.
                     E.g. "lambda x: sess.run(layer, feed_dict={ images_placeholder: x })".

    Stores:         
      recfield:  A np array of shape [feat_map_height, feat_map_width, 4],
                 where feat_map_height and feat_map_width are the dimensions
                 of the feature map. The array receptive_field[y,x,:]
                 has four chennels [y1, x1, y2, x2] corresponding to the
                 receptive field in the coordinates of input image.
    '''
    assert len(shape) == 4, shape
    assert all(v is not None for v in shape), \
      'Shape cannot be a placeholder, all four values must be defined.'
    [B, H, W, C] = shape
  
    # Compute vertical receptive field.
    batches = ReceptiveField._build_batches_vertical([B, H, W, C])
    logging.debug('Batches for computing vertical r.f. have shape %s' % str(batches.shape))
    maps = ReceptiveField._run_forward(batches, forward_func)
    logging.debug('Vertical features have shape %s' % str(maps.shape))
    y_recfield = ReceptiveField._interpret_vertical(maps)

    # Compute horizontal receptive field.
    batches = ReceptiveField._build_batches_vertical([B, W, H, C])
    batches = np.transpose(batches, axes=(0,1,3,2,4))
    logging.debug('Batches for computing horizontal r.f. have shape %s' % str(batches.shape))
    maps = ReceptiveField._run_forward(batches, forward_func)
    logging.debug('Horizontal features have shape %s' % str(maps.shape))
    maps = np.transpose(maps, axes=(0,2,1,3))
    x_recfield = ReceptiveField._interpret_vertical(maps)
    logging.debug('Horizontal intermediate r.f. shape %s' % str(x_recfield.shape))
    x_recfield = np.transpose(x_recfield, axes=(1,0,2))

    # combine vertical and horizontal.
    recfield = np.concatenate((y_recfield, x_recfield), axis=2)
    recfield = recfield[:,:,[0,2,1,3]]  # From [y1,y2,x1,x2] to [y1,x1,y2,x2].
    assert len(recfield.shape) == 3 and recfield.shape[2] == 4, recfield.shape
    np.testing.assert_array_less(recfield[:,:,0], recfield[:,:,2])
    np.testing.assert_array_less(recfield[:,:,1], recfield[:,:,3])
    np.set_printoptions(threshold=np.nan, linewidth=200)
    logging.info ('recfield shape: %s' % str(recfield.shape))
    logging.debug (np.transpose(recfield, (2,0,1)))

    self.imshape = shape
    self.recfield = recfield

  def max(self):
    ''' Find the maximum across receptive field of pixels in the feature maps.
    The receptive fields of different pixels differ mostly because of the
    location of these pixels near the feature map boundary.

    Returns:
                 A tuple (max_y, max_x) corresponding to maximum
                 receptive field in the two dimensions.
    '''
    return ((self.recfield[:,:,2] - self.recfield[:,:,0]).max(),
            (self.recfield[:,:,3] - self.recfield[:,:,1]).max())

  def draw_example(self):
    ''' Draw a receptive field for the central pixel in the feature map.

    Returns:
                 A grayscale image of shape [imshape[0], imshape[1]]
                 with a rectangle corresponding to the receptive field
                 of the central pixel in the feature map.
    '''
    y_feat = self.recfield.shape[0] // 2
    x_feat = self.recfield.shape[1] // 2
    y1, x1, y2, x2 = self.recfield[y_feat, x_feat].tolist()
    image = np.zeros(self.imshape[1:3], dtype=np.uint8)
    image[y1, x1:x2-1] = 255
    image[y2-1, x1:x2-1] = 255
    image[y1:y2-1, x1] = 255
    image[y1:y2-1, x2-1] = 255
    return image

