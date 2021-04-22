import tensorflow as tf
import utils
import ace05
import numpy as np
import itertools

DATA_PATH='./data/ace05/'

def get_feature_specs(nparray_dict):
  feature_specs = {}
  for key in nparray_dict.keys():
    feature_specs[key] =  tf.FixedLenFeature(shape=nparray_dict[key].shape,
                                             dtype=nparray_dict[key].dtype)
  return feature_specs

def pad_features(features, params):
  maxlen = params['max_len']
  new_features = {}
  for fea in features.keys():
    if fea == 'dep_path':
      continue
    if fea == 'has_dep': # not helping, not used
      continue
      num_example = len(features[fea])
      seq = np.zeros([num_example, maxlen, params['dep_size']])
      for i in xrange(num_example):
        for j,deps in itertools.izip(xrange(maxlen), features[fea][i]):
          for dep in deps:
            seq[i,j,dep] = 1.0
    elif fea == 'dep':
      seq = utils.pad_sequences(features[fea], 5, value=0)
    elif fea == 'pos1' or fea == 'pos2':
      seq = utils.pad_sequences(features[fea], maxlen, value=maxlen*2-1)
    elif fea == 'bag_size' or fea == 'mask' or fea == 'weight':
      seq = np.array(features[fea]) # no change
    else:
      seq = utils.pad_sequences(features[fea], maxlen, value=0)
    new_features[fea] = seq
  return new_features

def take_percentage(trainX, trainY, percentage):
  """take instances sequentially, assume input randomized
     remove the rest of instances"""
  if percentage == 1.0:
    return trainX, trainY
  new_num = int(len(trainY)*percentage)
  new_trainX = {}
  new_trainY = trainY[: new_num]
  for key in trainX.keys():
    new_trainX[key] = trainX[key][: new_num]
  print("number of training examples used: ", new_num)
  return new_trainX, new_trainY

def take_percentage_of_type_random(trainY, relation_type, percentage):
  """randomly selected
     mark the rest of instances as negative"""
  if percentage >= 1.0:
    return trainY
  trainY = np.argmax(trainY, axis=1)
  total_num = 0
  for i in xrange(len(trainY)):
    if trainY[i] == relation_type:
      if random.random() > percentage:
        trainY[i] = 0 # other
  return trainY

def take_percentage_of_type_miml(trainX, trainY, relation_type, percentage):
  if percentage >= 1.0:
    return trainX, trainY
  new_trainX, new_trainY = [], []
  total_num = 0
  for i in xrange(len(trainY)):
    if relation_type in trainY[i]:
      total_num += 1
  current_num = 0
  for i in xrange(len(trainY)):
    if relation_type in trainY[i]:
      current_num += 1
      if current_num < total_num * percentage:
        new_trainX += [trainX[i]]
        new_trainY += [trainY[i]]
    else:
        new_trainX += [trainX[i]]
        new_trainY += [trainY[i]]
  return new_trainX, new_trainY

def take_percentage_of_type(trainX, trainY, relation_type, percentage):
  """take instances squentially
  assume input shuffled
  remove the rest of instances from the dataset"""
  if percentage >= 1.0:
    return trainX, trainY
  nb_classes = trainY.shape[1]
  trainY = np.argmax(trainY, axis=1)
  total_num = 0
  for i in xrange(len(trainY)):
    if trainY[i] == relation_type:
      total_num += 1
  current_num = 0
  remove_list, retain_list = [], []
  for i in xrange(len(trainY)):
    if trainY[i] == relation_type:
      current_num += 1
      if current_num > total_num*percentage:
        remove_list += [i] # other
      else:
        retain_list += [i]
    else:
      retain_list += [i]
  trainY = utils.to_categorical(trainY, nb_classes=nb_classes)
  trainY = trainY[retain_list]
  for key in trainX.keys():
    trainX[key] = trainX[key][retain_list]
  return trainX, trainY

def pick_type_miml(trainY, relation_type):
  """trainY is a list of multi-label bags"""
  num = 0
  new_trainY = []
  for i in xrange(len(trainY)):
    if relation_type in trainY[i]:
      new_trainY += [[relation_type]] # ignore other types in the same bag
      num += 1
    else:
      new_trainY += [[0]] # other
  print('there are', num , ' relation_type', relation_type)
  return new_trainY
  
 
def pick_type(trainY, relation_types):
  """mark other types as negative
     if not in relation_types, 
     (warning) change index for positives"""
  nb_classes = trainY.shape[1]
  trainY = np.argmax(trainY, axis=1)
  for i in xrange(len(trainY)):
    if trainY[i] not in relation_types:
      trainY[i] = 0 # other
    else:
      trainY[i] = relation_types.index(trainY[i]) + 1
  trainY = utils.to_categorical(trainY, nb_classes=len(relation_types)+1) 
  return trainY 

def dict_input_producer(dict_tensors, num_epochs, shuffle=False):
  """input producer for a dictionary"""
  output= {} 
  output_values = []
  output_keys = []
  for key in dict_tensors.keys():
    output_keys += [key]
    output_values += [dict_tensors[key]]
  output_values = tf.train.slice_input_producer(
    output_values, shuffle=shuffle, num_epochs=num_epochs) 
#   print('producer', output_values)
  for i in range(len(output_keys)):
    output[output_keys[i]] = output_values[i]
  return output

# tf.estimator.inputs.numpy_input_fn
# TODO change to official version

def np_input_fn(data, batch_size, num_epoches, shuffle=False, eval=False):
  """data is a dict of numpy arrays (features)"""
  producer = dict_input_producer(data, num_epoches, shuffle)

  def input_fn():
    features = tf.train.batch(producer, batch_size=batch_size, 
                              allow_smaller_final_batch=True if eval else False
                              )
    targets = features.pop('y')
    targets = tf.to_float(targets)
    return features, targets
  return input_fn
