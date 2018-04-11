import numpy as np
import cv2


class Activations():
  def __init__(self, num_samples=10):
    self.images = None
    self.firings = None
    self.coords = None
    self.Size = num_samples

  def _multiple_axes_max(self, x):
    max_val = x.reshape(x.shape[0], -1).max(axis=1)
    max_idx = x.reshape(x.shape[0], -1).argmax(axis=1)
    maxpos_vect = np.column_stack(np.unravel_index(max_idx, x[0].shape))
    return max_val, maxpos_vect

  def update(self, images, features):
    assert len(images.shape) == 4
    assert len(features.shape) == 3  # Each only one filter.
    assert images.shape[0] == features.shape[0]
    # Extract the highest activation value and its coords for each image.
    firings, coords = self._multiple_axes_max(features)
    assert len(coords.shape) == 2, coords.shape
    assert coords.shape[0] == features.shape[0], coords.shape
    if self.images is None:
      self.images = images
      self.firings = firings
      self.coords = coords
    else:
      self.images = np.concatenate((self.images, images), axis=0)
      self.firings = np.concatenate((self.firings, firings), axis=0)
      self.coords = np.concatenate((self.coords, coords), axis=0)
    # Pick the images with highest activations.
    best_indices = np.argsort(-self.firings)
    self.images = self.images[best_indices,:,:,:]
    self.firings = self.firings[best_indices]
    self.coords = self.coords[best_indices]
    # Limit stored values to Size.
    self.images = self.images[:self.Size]
    self.firings = self.firings[:self.Size]
    self.coords = self.coords[:self.Size]
    
  def retrieve(self, recfield):
    maxsize, maxsize = recfield.max()
    images = self.images.copy()
    crops = []
    # Draw a rectangle.
    for (image, coord) in zip(images, self.coords):
      feat_y, feat_x = coord[0], coord[1]
      y1, x1, y2, x2 = recfield.recfield[feat_y, feat_x].tolist()
      # Crop.
      #print ('[%dx%d]->[%dx%d %dx%d] ' % (feat_y, feat_x, y1, x1, y2, x2), end='')
      crop = image[y1:y2, x1:x2, :]
      # Add black if the crop is on the border.
      pad_w = maxsize - crop.shape[0]
      pad_h = maxsize - crop.shape[1]
      crop = np.pad(crop, ((pad_w // 2, pad_w - pad_w // 2),
                           (pad_h // 2, pad_h - pad_h // 2),
                           (0, 0)), 'constant')
      # Add a thin black boundary.
      pad = maxsize // 5
      crop = np.pad(crop, ((1,1),(pad,pad),(0,0)), 'constant')
      crops.append(crop)
      # Draw a rectangle on the image.
      cv2.rectangle(image, (x1-2, y1-2), (x2+2-1, y2+2-1), (0,0,255), 2)
      # Add a thin black boundary.
      pad = image.shape[1] // 5
      image = np.pad(image, ((1,1),(pad,pad),(0,0)), 'constant')
    #print ('.')
    shape = images.shape
    images = images.reshape((shape[0]*shape[1], shape[2], shape[3]))
    crops = np.stack(crops)
    shape = crops.shape
    crops = crops.reshape((shape[0]*shape[1], shape[2], shape[3]))
    return images, crops

