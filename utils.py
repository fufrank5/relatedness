from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _prod(array):
  prod = 1.
  for e in array:
    prod *= e
  return prod


def variable_shapes(optimized_vars=None):
  lines = ['']
  if optimized_vars is not None:
    lines.append('Optimizing Variables:')
    lines.append('====================')
    total_params = 0
    for var in optimized_vars:
      n_param = _prod(var.get_shape().as_list())
      total_params += n_param
      lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
    lines.append('Total Optimizing parameters: %d' % total_params)

    lines.append('')

  train_vars = tf.trainable_variables()
  lines.append('Trainable Variables:')
  lines.append('====================')
  total_params = 0
  for var in train_vars:
    if optimized_vars is not None and var in optimized_vars: continue
    n_param = _prod(var.get_shape().as_list())
    total_params += n_param
    lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
  lines.append('Total trainable parameters: %d' % total_params)

  lines.append('')
  lines.append('Other Variables:')
  lines.append('================')
  total_params = 0

  for var in tf.global_variables():
    if var in train_vars: continue
    n_param = _prod(var.get_shape().as_list())
    total_params += n_param
    lines.append('%20s %8d %s' % (var.get_shape().as_list(), n_param, var.name))
  lines.append('Total non-trainable parameters: %d' % total_params)

  return '\n'.join(lines)


def print_attention_weights(features, targets, mode, params, review_weights, predictions):
  if review_weights is NOne:
    tf.logging.info('no review_weights')
    return predictions

  if mode != tf.contrib.learn.ModeKeys.TRAIN:
    print_num = params['print_num_per_batch']
    max_weight_index = tf.argmax(review_weights, axis=1)
    max_weightindex = tf.squeeze(max_weight_index, axis=1)
    review_weights = tf.squeeze(review_weights, axis=2)
    for i in range(print_num):
      predictions = tf.Print(predictions, [features['placeid'][i]], summarize=1)
      predictions = tf.Print(predictions, [features['attribute_string'][i]], summarize=1)
      predictions = tf.Print(predictions, [max_weight_index[i]], summarize=1)
      predictions = tf.Print(predictions, [review_weights[i, max_weight_index[i]]], summarize=1)
      predictions = tf.Print(predictions, [features['token_list_string'][i, max_weight_index[i]]], summarize=params['max_len'])
      predictions = tf.Print(predictions, [predictions[i]], summarize=1)

      if mode == tf.contrib.learn.ModeKeys.EVAL:
        predictions = tf.Print(predictions, [targets[i]], summarize=1)

  return predictions


def to_categorical(y, nb_classes, miml=False):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

    """
    Y = np.zeros((len(y), nb_classes), dtype=np.float32)
    for i in range(len(y)):
      if miml:
        for label in y[i]:
          Y[i, label] = 1.
      else:
        Y[i, y[i]] = 1.
    return Y


# =====================
#    SEQUENCES UTILS
# =====================


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.

    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)

    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def load_word2vec(fname, vocab, lower_case=False):
  """
  Loads 300x1 word vecs from Google (Mikolov) word2vec
  """
  word_vecs = {}
  with open(fname, "rb") as f:
      header = f.readline()
      vocab_size, layer1_size = map(int, header.split())
      binary_len = np.dtype('float32').itemsize * layer1_size
      for line in xrange(vocab_size):
          word = []
          while True:
              ch = f.read(1)
              if ch == ' ':
                  word = ''.join(word)
                  break
              if ch != '\n':
                  word.append(ch) 
          if lower_case:
            word = word.lower()  
          if word in vocab:
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
          else:
            f.read(binary_len)
  return word_vecs
