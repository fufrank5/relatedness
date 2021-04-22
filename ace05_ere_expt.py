from __future__ import print_function

import pprint
import math
import time, datetime

import numpy as np
import tensorflow as tf
import ace05, ere
from utils import to_categorical
import utils
import random
from inputs import pad_features, take_percentage
import inputs
import metrics
from joint_models import pretrain_relation_model, joint_stack_model, joint_relation_model, joint_transfer_model, joint_separate_model, joint_auxiliary_model, joint_meta_model, joint_bi_rnn_model, \
joint_repr_model, joint_kshot_model, joint_ksupport_model

MODEL_FN = {
  'joint': joint_relation_model,
  'joint_ere2ace': joint_transfer_model, 
  'joint_ace2ere' : joint_transfer_model,
  'joint_aux': joint_auxiliary_model,
  'joint_separate': joint_separate_model,
  'joint_stack': joint_stack_model,
  'joint_bi_rnn': joint_bi_rnn_model,
  'joint_repr': joint_repr_model,
  'joint_repr_NONE': joint_repr_model,
  'joint_reader': joint_repr_model,
  'joint_embedding': joint_repr_model,
  'joint_kshot': joint_kshot_model,
  'joint_ksupport': joint_ksupport_model,
}


def preprocess_ace05_all(datasets, params):
  train, dev, test = datasets
  train = preprocess_ace05(train, params)
  def shuffle_ace05_train(train):
    trainX, trainY = train
    new_trainX = {}
    new_trainY = []
    indices = range(len(trainY))
    random.shuffle(indices)
    for key in trainX.keys():
      new_trainX[key] = []
      for i in indices:
        new_trainX[key] += [trainX[key][i]]
    for i in indices:
      new_trainY += [trainY[i]]
    return trainX, trainY
   
#  train = shuffle_ace05_train(train)
  dev = preprocess_ace05(dev, params)
  processed_test = {}
  for corpus in test:
    processed_test[corpus] = preprocess_ace05(test[corpus], params)
  return train, dev, processed_test

def preprocess_ace05(dataset, params):
  X, Y = dataset
  X = pad_features(X, params)
  Y = to_categorical(Y, nb_classes=params['ace05_num_classes'])
  return X, Y


def read_ace05(params):
  ace05_data, embed = ace05.load_dataset(max_len=params['max_len'], directional=params['directional'])
  if not params['directional']:
    params['ace05_num_classes'] = 7
  entity_type, vocab, label_set = ace05.read_entity_vocab(directional=params['directional'])
  ace05_data = preprocess_ace05_all(ace05_data, params) # train, dev, test
  print('ace entity', entity_type)
  return ace05_data, entity_type, vocab, label_set, embed

def prepare_ace05(ace05_data, params):
  trainX = ace05_data[0][0]
  trainY = ace05_data[0][1]
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x=trainX, y=trainY, 
      num_epochs=None, batch_size=params['batch_size'], shuffle=True)
  num_steps_per_epoch = trainY.shape[0]/params['batch_size'] + 1
  params['learning_rate_decay_step'] = int(num_steps_per_epoch*params['lr_decay_epoch'])
  print('num steps per epoch for ace05', num_steps_per_epoch)
  config = tf.estimator.RunConfig()
  config = config.replace(tf_random_seed=params['random_seed'], keep_checkpoint_max=1)
                          #save_checkpoints_stepsi=num_steps_per_epoch)
  estimator = tf.estimator.Estimator(model_dir=params['model_dir']+'/ace/', model_fn=pretrain_relation_model, 
                                     params=params,
                                     config=config)
  evaluator = metrics.EvaluatorACE05Hook(estimator, ace05_data)
  return estimator, train_input_fn, num_steps_per_epoch, evaluator

def take_kshot(X, Y, k, all_other=True):
  """return instances"""
  indexY = np.argmax(Y, axis=1)
  nb_classes = Y.shape[1]
  num_pos = (nb_classes-1)*k
  num_pos_total = np.count_nonzero(indexY)
  num_neg = (len(indexY) - num_pos_total)*num_pos/num_pos_total
  k_list = []
  for y in range(0, nb_classes):
    positive_list = []
    for i in xrange(len(indexY)):
      if indexY[i] == y:
        positive_list += [i]
    if all_other and y == 0:
      k_list += positive_list
    elif y == 0:
      k_list += random.sample(positive_list, num_neg)
    else:
      k_list += random.sample(positive_list, k)
  kshot_Y = Y[k_list]
  kshot_X = {}
  for key in X.keys():
    kshot_X[key] = X[key][k_list]
  
  return kshot_X, kshot_Y

def prepare_kshot(X, Y, nb_classes, k, only_k=False):
  """return input_fns"""
  indexY = np.argmax(Y, axis=1)
  input_fns = []
  for y in range(0, nb_classes):
    positive_list = []
    for i in xrange(len(indexY)):
      if indexY[i] == y:
        positive_list += [i]
    if only_k:
      if k < len(positive_list):
        positve_list = random.sample(positive_list, k)
    selectY = Y[positive_list]
    selectX = {}
    for key in X.keys():
      selectX[key] = X[key][positive_list]
    input_fn = tf.estimator.inputs.numpy_input_fn(x=selectX, y=selectY, 
      num_epochs=None, batch_size=k, shuffle=True)
    input_fns += [input_fn]
  return input_fns

def prepare_exclusive_kshot(X, Y, nb_classes, k, batch_size, eval=False):
  indexY = np.argmax(Y, axis=1)
  indices = []
  for y in range(0, nb_classes):
    indices += [[]]
  for i in xrange(len(indexY)):
    indices[indexY[i]] += [i]
  indices_set = set(range(len(indexY)))

  max_num = len(indices_set)
  num_steps = int(math.ceil(max_num/float(batch_size))) # num of steps for one epoch

  # for each step
  data = {}
  for key in X.keys():
    data['support_' +key] = []
    data['query_' + key] = []
  data['support_y'] = []
  data['query_y'] = []
  for i in range(num_steps):
    support_indices = []
    for y in range(0, nb_classes):
      support_indices += random.sample(indices[y],k)
    support_indices_set = set(support_indices)
    if len(indices_set-support_indices_set) < batch_size:
      query_indices = list(indices_set-support_indices_set)
    else:
      query_indices = random.sample(indices_set-support_indices_set, batch_size)
    indices_set = indices_set - set(query_indices) # no replacement
    for key in X.keys():
      data['support_' + key] += [X[key][support_indices]]
    data['support_y'] += [Y[support_indices]]
    for key in X.keys():
      data['query_' + key] += [X[key][query_indices]]
    data['query_y'] += [Y[query_indices]]
  for key in data.keys():
    data[key] = np.stack(data[key], axis=0)
  return data

def kshot_input_fn(data, num_epochs, shuffle=False):
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x=trainX, y=trainY, 
      num_epochs=None, batch_size=params['batch_size'], shuffle=True)


def preprocess_ere(data, params):
  # change list of dictionary to dictionary of lists
  data = dict(zip(data[0],zip(*[d.values() for d in data])))
  Y = data.pop('y')
  X = data
  X = pad_features(X, params)
  Y = to_categorical(Y, nb_classes=params['ere_num_classes'])
  return X, Y

def read_ere(params):
  ere_split = {
    '0.8': [0.8, 0.1 ,0.1],
    '0.4': [0.4, 0.2, 0.4]
  }
  train, dev, test, vocab, label_set, word_embed = ere.load_dataset(
      vocab_size=params['ere_vocab_size'], max_len=params['max_len'], include_subtype=params['subtype'],
      split=ere_split[params['ere_split']])
  if params['subtype']:
    params['ere_num_classes'] = 19
  entity_set = ere.load_entity_set()
  print('ere entity', entity_set)
  random.shuffle(train)
  trainX, trainY = preprocess_ere(train, params)
  devX, devY = preprocess_ere(dev, params)
  testX, testY = preprocess_ere(test, params)
  ere_data = ((trainX, trainY), (devX, devY), (testX, testY))
  return ere_data, entity_set, vocab, label_set, word_embed

def prepare_ere(ere_data, params):
  trainX = ere_data[0][0]
  trainY = ere_data[0][1]
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x=trainX, y=trainY, 
      num_epochs=None, batch_size=params['batch_size'], shuffle=True)
  num_steps_per_epoch = trainY.shape[0]/params['batch_size'] + 1
  params['learning_rate_decay_step'] = int(num_steps_per_epoch*params['lr_decay_epoch'])
  print('num steps per epoch for ere', num_steps_per_epoch)
  config = tf.estimator.RunConfig()
  config = config.replace(tf_random_seed=params['random_seed'], keep_checkpoint_max=1)
                          #save_checkpoints_steps=num_steps_per_epoch)
  estimator = tf.estimator.Estimator(model_dir=params['model_dir']+'/ere', model_fn=pretrain_relation_model, 
                                     params=params,
                                     config=config)
  evaluator = metrics.EvaluatorEREHook(
      estimator, ere_data)
  return estimator, train_input_fn, num_steps_per_epoch, evaluator


def translate_entity_type(relations, from_entity_set, to_entity_set):
  """relations has to be a dict of, 
  entity_set is entity type to index mapping
  """
  translator = {0:0} # for padding 0
  for e_type, index in from_entity_set.items():
    if e_type not in to_entity_set:
      if e_type  == 'Other': # ace use Other
        e_type = 'O' # ere use O
      else:
        e_type = 'Other'
    translator[index] = to_entity_set[e_type]
  for r in relations['type']:
    for i in xrange(len(r)):
      r[i] = translator[r[i]]

def translate_vocab_index(relations, from_vocab, to_vocab):
  """relations has to be a list of dict, 
  vocab is word to index mapping
  """
  translator = {0:0} # </s>
  for word, index in from_vocab.items():
    translator[index] = to_vocab[word]
  for r in relations['word']:
    for i in xrange(len(r)):
      r[i] = translator[r[i]]

def concat_vocab(vocab1, vocab2, embed1, embed2):
# vocab1 unchanged
  vocab = dict(vocab1)
  count = len(vocab)
  indices = []
  for word, index in vocab2.items():
    if word not in vocab:
      vocab[word] = count
      count += 1
      indices += [index]
  embed = np.concatenate((embed1, embed2[indices, :]), axis=0)
  return vocab, embed


def make_kshot_input_fn(ace05_fn, ace05_kshot_fns, ere_fn, ere_kshot_fns):
  def input_fn():
    ace05_X, ace05_Y = ace05_fn()
    ere_X, ere_Y = ere_fn()

    features, targets = {}, {}
    for key in ace05_X.keys():
      features['ace05_' + key] = ace05_X[key]
    for key in ere_X.keys():
      features['ere_' + key] = ere_X[key]
    targets['ace05_relation'] = ace05_Y
    targets['ere_relation'] = ere_Y

    #ace05 kshot fns
    targets['ace05_kshot_relation'] = []
    for i in range(len(ace05_kshot_fns)):
      ace05_kshot = ace05_kshot_fns[i]
      ace05_kshot_X, ace05_kshot_Y = ace05_kshot()

      for key in ace05_kshot_X.keys():
        if 'ace05_kshot_' + key not in features:
          features['ace05_kshot_' + key] = []
        features['ace05_kshot_' + key] += [ace05_kshot_X[key]]
      targets['ace05_kshot_relation'] += [ace05_kshot_Y]
    for key in ace05_kshot_X.keys():
      features['ace05_kshot_' + key] = tf.concat(features['ace05_kshot_' + key], axis=0)
    targets['ace05_kshot_relation'] = tf.concat(targets['ace05_kshot_relation'], axis=0)

    #ere kshot fns
    targets['ere_kshot_relation'] = []

    for i in range(len(ere_kshot_fns)):
      ere_kshot = ere_kshot_fns[i]
      ere_kshot_X,ere_kshot_Y = ere_kshot()
    
      for key in ere_kshot_X.keys():
        if 'ere_kshot_' + key not in features:
          features['ere_kshot_' + key] = []
        features['ere_kshot_' + key] += [ere_kshot_X[key]]
      targets['ere_kshot_relation'] += [ere_kshot_Y]
    for key in ere_kshot_X.keys():
      features['ere_kshot_' + key] = tf.concat(features['ere_kshot_' + key], axis=0)
    targets['ere_kshot_relation'] = tf.concat(targets['ere_kshot_relation'], axis=0)

    return features, targets
  return input_fn

def make_two_input_fn(ace05_fn, ace05_steps, ere_fn, ere_steps):
  def input_fn():
    ace05_X, ace05_Y = ace05_fn()
    ere_X, ere_Y = ere_fn()
    features, targets = {}, {}
    for key in ace05_X.keys():
      features['ace05_' + key] = ace05_X[key]
    for key in ere_X.keys():
      features['ere_' + key] = ere_X[key]
    targets['ace05_relation'] = ace05_Y
    targets['ere_relation'] = ere_Y

    return features, targets
  return input_fn

def evaluate_ere_test(estimator, evaluator, test, params, output_types=None):
  estimator._model_dir = evaluator.best_ckpt
  test_score = metrics.evaluate_dataset(estimator, test, params['ere_num_classes'], output_types=output_types)
  with open(params['score_file'] + '_ere','a') as score_output:
    print('dev %4.2f' % (100.0*evaluator.best_dev.item()), end=' ')
    print('dev %4.2f' % (100.0*evaluator.best_dev.item()), end=' ', file=score_output)
    print('test %4.2f' % (100.0*test_score.item()), end=' ')
    print('test %4.2f' % (100.0*test_score.item()), end=' ', file=score_output)
    
    print('epoch', evaluator.best_epoch, end=' ')
    print('epoch', evaluator.best_epoch, end=' ', file=score_output)
    print('dir', evaluator.best_ckpt, file=score_output)

def evaluate_ace05_test(estimator, evaluator, test, params):
  estimator._model_dir = evaluator.best_ckpt
  with open(params['score_file']+'_ace','a') as score_output:
    print('bc0 %4.2f' % (100.0*evaluator.best_dev.item()), end=' ')
    print('bc0 %4.2f' % (100.0*evaluator.best_dev.item()), end=' ', file=score_output)
    for corpus in test:
      score = metrics.evaluate_dataset(estimator, test[corpus], params['ace05_num_classes'])
      print(corpus,'%4.2f' % (100.0*score.item()), end=' ')
      print(corpus,'%4.2f' % (100.0*score.item()), end=' ', file=score_output)
    print('epoch', evaluator.best_epoch, end=' ')
    print('epoch', evaluator.best_epoch, end=' ', file=score_output)
    print('dir', evaluator.best_ckpt, file=score_output)
    print()

def evaluate_joint_ere_test(estimator, evaluator, test, ace05_test, params, ere_label_list=None, output_types=None,
                            ace05_kshot_data=None, ere_kshot_data=None):
  if params['kshot']:
    test_input_fn = make_joint_kshot_eval_inputs(ace05_test, ace05_kshot_data, test, ere_kshot_data, params)
  else:
    test_input_fn = make_joint_eval_inputs(ace05_test, test)
  test_score, preds = evaluator.evaluate_dataset(estimator, test_input_fn, test[1], params['ere_num_classes'], 'ere_relation', output_types=output_types)
  all_scores = metrics.f1_metric(test[1], preds, params['ere_num_classes'], average=None)
  print(all_scores)
  with open(params['score_file'] + '_ere','a') as score_output:
    print('dev %4.2f' % (100.0*evaluator.best_dev), end=' ')
    print('dev %4.2f' % (100.0*evaluator.best_dev), end=' ', file=score_output)
    print('test %4.2f' % (100.0*test_score), end=' ')
    print('test %4.2f' % (100.0*test_score), end=' ', file=score_output)
    if ere_label_list is not None:
      for i in xrange(len(all_scores)):
        print(str(ere_label_list[i])+ ' %4.2f' % (100.0*all_scores[i]), end=' ')
        print(str(ere_label_list[i])+ ' %4.2f' % (100.0*all_scores[i]), end=' ', file=score_output)
    print('epoch', evaluator.best_epoch, end=' ')
    print('epoch', evaluator.best_epoch, end=' ', file=score_output)
    print('dir', evaluator.best_ckpt, file=score_output)
    print()

def get_list_by_key(pred_iter, key):
  preds = []
  for pred in pred_iter:
    preds += [pred[key]]
  preds = np.argmax(preds, 1)
  return preds

def predict_joint_test(estimator, test, corpus, params):
  test_input_fn = make_joint_eval_inputs_mock(test)
  output = estimator.predict(input_fn=test_input_fn, predict_keys=['ace05_relation', 'ere_relation'])
  output = list(output)
  ace05_output = get_list_by_key(output, 'ace05_relation')
  ere_output = get_list_by_key(output, 'ere_relation')
  test_output = np.argmax(test[1], 1)
  with open(params['output_file'] + '_' + corpus, 'w') as pred_output:
    for p1, p2, t in zip(ace05_output, ere_output, test_output):
      print(p1, p2, t, file=pred_output)


def evaluate_joint_ace05_test(estimator, evaluator, test, ere_test, params,
                              ace05_kshot_data=None, ere_kshot_data=None):
  # for joint model
  with open(params['score_file']+'_ace','a') as score_output:
    print('bc0 %4.2f' % (100.0*evaluator.best_dev), end=' ')
    print('bc0 %4.2f' % (100.0*evaluator.best_dev), end=' ', file=score_output)
    all_preds, all_truth = [], []
    macro_avg = 0.0
    for corpus in test:
      if params['kshot']:
        test_input_fn = make_joint_kshot_eval_inputs(test[corpus], ace05_kshot_data, ere_test, ere_kshot_data, params)
      else:
        test_input_fn = make_joint_eval_inputs(test[corpus], ere_test)
      score, preds = evaluator.evaluate_dataset(estimator, test_input_fn, test[corpus][1], params['ace05_num_classes'], 'ace05_relation')
      all_preds += [preds]
      all_truth += [test[corpus][1]]
      macro_avg += score
      print(corpus,'%4.2f' % (100.0*score), end=' ')
      print(corpus,'%4.2f' % (100.0*score), end=' ', file=score_output)
    all_preds = np.concatenate(all_preds, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)
    micro_avg = metrics.f1_metric(all_truth, all_preds, params['ace05_num_classes'])
    macro_avg /= float(len(test))
    print('micro', '%4.2f' % (100.0*micro_avg), end=' ' )
    print('micro', '%4.2f' % (100.0*micro_avg), end=' ', file=score_output)
    print('macro', '%4.2f' % (100.0*macro_avg), end=' ' )
    print('macro', '%4.2f' % (100.0*macro_avg), end=' ', file=score_output)
    print('epoch', evaluator.best_epoch, end=' ')
    print('epoch', evaluator.best_epoch, end=' ', file=score_output)
    print('dir', evaluator.best_ckpt, file=score_output)

def make_joint_eval_inputs_mock(data):
  input_fn = tf.estimator.inputs.numpy_input_fn(x=data[0], y=data[1], num_epochs=1, shuffle=False)
  return make_two_input_fn(input_fn, input_fn)

def make_joint_eval_inputs(ace05_data, ere_data): # for joint model
  ace05_input_fn=  tf.estimator.inputs.numpy_input_fn(x=ace05_data[0], y=ace05_data[1], num_epochs=1, shuffle=False)
  ere_input_fn =  tf.estimator.inputs.numpy_input_fn(x=ere_data[0], y=ere_data[1], num_epochs=1, shuffle=False) # to cope with the format
  return make_two_input_fn(ace05_input_fn, None, ere_input_fn, None)

def make_joint_kshot_eval_inputs(ace05_data, ace05_kshot_data, ere_data, ere_kshot_data, params):
  ace05_train_kshot_eval_fn = prepare_kshot(ace05_kshot_data[0], ace05_kshot_data[1], params['ace05_num_classes'], params['ksupport_size'], only_k=True)
  ere_train_kshot_eval_fn = prepare_kshot(ere_kshot_data[0], ere_kshot_data[1], params['ere_num_classes'], params['ksupport_size'], only_k=True)
  ace05_eval_fn = tf.estimator.inputs.numpy_input_fn(x=ace05_data[0], y=ace05_data[1], num_epochs=1, shuffle=False)
  ere_eval_fn = tf.estimator.inputs.numpy_input_fn(x=ere_data[0], y=ere_data[1], num_epochs=1, shuffle=False)
  ace05_train_eval_fn = make_kshot_input_fn(ace05_eval_fn, ace05_train_kshot_eval_fn, ere_eval_fn, ere_train_kshot_eval_fn)
  return ace05_train_eval_fn

def filter_training_data(ace05_data, ere_data, params):
  """reduce training data according to percentage or kshot"""
  trainX = ace05_data[0][0]
  trainY = ace05_data[0][1]
  trainX, trainY = take_percentage(trainX, trainY, params['ace05_percent_train'])
  if params['ace05_kshot'] != 0:
    trainX, trainY = take_kshot(trainX, trainY, params['ace05_kshot'], all_other=False)
    print('ace05_kshot', len(trainY))
  ace05_data = ((trainX, trainY), ace05_data[1], ace05_data[2])

  trainX = ere_data[0][0]
  trainY = ere_data[0][1]
  trainX, trainY = take_percentage(trainX, trainY, params['ere_percent_train'])
  if params['ere_kshot'] != 0:
    trainX, trainY = take_kshot(trainX, trainY, params['ere_kshot'], all_other=False)
  ere_data = ((trainX, trainY), ere_data[1], ere_data[2])
  return ace05_data, ere_data

def determine_batch_size_lamda(ace05_data, ere_data, params):
  if ere_data[0][1].shape[0] > ace05_data[0][1].shape[0]:
    ere_batch_size = params['batch_size']
    ace05_batch_size = int(math.ceil(params['batch_size'] * ace05_data[0][1].shape[0] / float(ere_data[0][1].shape[0])))
  else:
    ace05_batch_size = params['batch_size']
    ere_batch_size = int(math.ceil(params['batch_size'] * ere_data[0][1].shape[0] / float(ace05_data[0][1].shape[0])))

  lamda_base = ere_batch_size/float(ace05_batch_size + ere_batch_size)
  print('base lamda', lamda_base)
  if params['dynamic_lamda']:
    lamda_base = params['lamda']*lamda_base
  else:
    lamda_base = params['lamda']
  print('adjusted lamda', lamda_base)
  print('batch_size for ace05', ace05_batch_size)
  print('batch_size for ere',ere_batch_size)   
  return ace05_batch_size, ere_batch_size, lamda_base


def run_joint(ace05_data, ere_data, params, ace05_label_set=None, ere_label_set=None):
  params['percent_train'] = params['ace05_percent_train']
  ace05_data, ere_data = filter_training_data(ace05_data, ere_data, params)
  ace05_batch_size, ere_batch_size, lamda = determine_batch_size_lamda(ace05_data, ere_data, params)
  params['lamda'] = lamda
  params['batch_size'] = ace05_batch_size
  ace05_estimator, ace05_input_fn, ace05_steps, ace05_evaluator = prepare_ace05(ace05_data, params)
  params['batch_size'] = ere_batch_size
  ere_estimator, ere_input_fn, ere_steps, ere_evaluator = prepare_ere(ere_data, params)
  print(ace05_steps, ere_steps)
  steps=ace05_steps
  params['learning_rate_decay_step'] = int(steps*params['lr_decay_epoch'])
  print('num steps per epoch', steps)
  train_input_fn = make_two_input_fn(ace05_input_fn, ace05_steps, ere_input_fn, ere_steps)
  config = tf.estimator.RunConfig(save_checkpoints_steps=params['eval_every_n_epoch']*steps, tf_random_seed=params['random_seed'], keep_checkpoint_max=1)
  model_fn = MODEL_FN[params['joint']]
  joint_estimator = tf.estimator.Estimator(model_dir=params['model_dir'], model_fn=model_fn, params=params, config=config)
  ace05_train_eval_fn = make_joint_eval_inputs(ace05_data[0], ere_data[0])
  ace05_dev_input_fn = make_joint_eval_inputs(ace05_data[1], ere_data[1])
  ere_train_eval_fn = make_joint_eval_inputs(ace05_data[0], ere_data[0])
  ere_dev_input_fn = make_joint_eval_inputs(ace05_data[1], ere_data[1])

    
  if params['target_type'] != '':
    output_types=[ere_label_set[params['target_type']]]
  else:
    output_types=None
  

  if params['kshot']:
#    ace05_kshot_data = prepare_exclusive_kshot(ace05_data[0][0], ace05_data[0][1], params['ace05_num_classes'], params['ksupport_size'], params['batch_size'])
#    print(ace05_kshot_data)
    ace05_kshot_input_fn = prepare_kshot(ace05_data[0][0], ace05_data[0][1], params['ace05_num_classes'], params['ksupport_size'])
    ere_kshot_input_fn = prepare_kshot(ere_data[0][0], ere_data[0][1], params['ere_num_classes'], params['ksupport_size'])

    train_input_fn = make_kshot_input_fn(ace05_input_fn, ace05_kshot_input_fn, ere_input_fn, ere_kshot_input_fn)
    train_eval_fn = make_joint_kshot_eval_inputs(ace05_data[0], ace05_data[0], ere_data[0], ere_data[0], params)
    dev_eval_fn = make_joint_kshot_eval_inputs(ace05_data[1], ace05_data[0], ere_data[1], ere_data[0], params)
    
    evaluator = metrics.EvaluatorACE05EREHook(joint_estimator, steps,
        train_eval_fn, dev_eval_fn, ace05_data[0][1], ace05_data[1][1], params['ace05_num_classes'],
        train_eval_fn, dev_eval_fn, ere_data[0][1], ere_data[1][1], params['ere_num_classes'],
        output_types=output_types) # target_type for ERE only
  else:
    evaluator = metrics.EvaluatorACE05EREHook(joint_estimator, steps,
        ace05_train_eval_fn, ace05_dev_input_fn, ace05_data[0][1], ace05_data[1][1], params['ace05_num_classes'],
        ere_train_eval_fn, ere_dev_input_fn, ere_data[0][1], ere_data[1][1], params['ere_num_classes'],
        output_types=output_types) # target_type for ERE only
  
  if params['debug']:
    print('debug mode')
    steps = 100
    params['epoch'] = 1

  if params['predict']:
    joint_estimator = tf.estimator.Estimator(
        model_dir=params['predict_model_dir']+'/ace05_relation_best/',
        model_fn=model_fn, params=params, config=config)
    evaluate_joint_ace05_test(joint_estimator, evaluator.ace05_evaluator, ace05_data[2], ere_data[2], params)
    predict_joint_test(joint_estimator, ace05_data[2]['wl'], 'ace05_wl', params)
    predict_joint_test(joint_estimator, ace05_data[2]['bc1'], 'ace05_bc1', params)
    predict_joint_test(joint_estimator, ace05_data[2]['cts'], 'ace05_cts', params)
    joint_estimator._model_dir = params['predict_model_dir']+'/ere_relation_best/'
    evaluate_joint_ere_test(joint_estimator, evaluator.ere_evaluator, ere_data[2], ace05_data[2]['bc1'], params)
    predict_joint_test(joint_estimator, ere_data[2], 'ere', params)
  else: # Train and Eval
    for epoch in range(1, params['epoch']+1):
      print('==========')
      print('epoch', epoch)
      joint_estimator._params['current_epoch'] = epoch
      joint_estimator.train(input_fn=train_input_fn,
                      steps=steps,
                      hooks=[evaluator])
    print('finish training ace, best dev (%4.4f) found at epoch: %s'
           % (evaluator.ace05_evaluator.best_dev, str(evaluator.ace05_evaluator.best_epoch)))
    print('finish training ere, best dev (%4.4f) found at epoch: %s'
           % (evaluator.ere_evaluator.best_dev, str(evaluator.ere_evaluator.best_epoch)))
    joint_estimator._model_dir = evaluator.ace05_evaluator.best_ckpt
    evaluate_joint_ace05_test(joint_estimator, evaluator.ace05_evaluator, ace05_data[2], ere_data[2], params, ace05_data[0], ere_data[0])
    joint_estimator._model_dir = evaluator.ere_evaluator.best_ckpt
    ere_label_list = [''] * len(ere_label_set)
    for label, index in ere_label_set.iteritems():
      if index == 0: continue
      ere_label_list[index-1] = label
    if params['target_type'] == '':
      evaluate_joint_ere_test(joint_estimator, evaluator.ere_evaluator, ere_data[2], ace05_data[2]['bc1'], params, 
                              ace05_kshot_data=ace05_data[0], ere_kshot_data=ere_data[0])
    else:
      evaluate_joint_ere_test(joint_estimator, evaluator.ere_evaluator, ere_data[2], ace05_data[2]['bc1'], params,
          ere_label_list=ere_label_list, output_types=[ere_label_set[params['target_type']]],
          ace05_kshot_data=ace05_data[0], ere_kshot_data=ere_data[0])


    # evaluate_joint_ere_test(joint_estimator, evaluator.ere_evaluator, ere_data[2], ace05_data[2]['bc1'], params,
    #     ere_label_list=ere_label_list)

  return evaluator.ace05_evaluator.best_ckpt, evaluator.ere_evaluator.best_ckpt  


def run_ace2ere(ace05_data, ere_data, params):
  percent = params['percent_train']
  params['percent_train'] = 1.0
  if not params['use_existing_pretrain']:
    best_ckpt = run_ace(ace05_data, params)
    params['pretrained_model'] = best_ckpt
  else:
    params['pretrained_model'] = './pretrained_models/ace05/'
  params['percent_train'] = percent
  run_ere(ere_data, params)


def run_ere2ace(ere_data, ace05_data, params):
  percent = params['percent_train']
  params['percent_train'] = 1.0
  if not params['use_existing_pretrain']:
    best_ckpt = run_ere(ere_data, params)
    params['pretrained_model'] = best_ckpt
  else:
    params['pretrained_model'] = './pretrained_models/ere/'
  params['percent_train'] = percent
  run_ace(ace05_data, params)

def run_ace(ace05_data, params):
  params['current_dataset'] = 'ace05'
  ace05_estimator, ace05_input_fn, ace05_steps, ace05_evaluator = prepare_ace05(ace05_data, params)
  if params['debug']:
    print('debug mode')
    ace05_steps = 100
    params['epoch'] = 1
  for epoch in range(1, params['epoch']+1):
    print('==========')
    print('epoch', epoch)
    ace05_estimator._params['current_epoch'] = epoch 
    ace05_estimator.train(input_fn=ace05_input_fn,
                    steps=ace05_steps,
                    hooks=[ace05_evaluator])
  print('finish training ace, best dev (%4.4f) found at epoch: %d'
         % (ace05_evaluator.best_dev, ace05_evaluator.best_epoch))
  evaluate_ace05_test(ace05_estimator, ace05_evaluator, ace05_data[2], params)
  return ace05_evaluator.best_ckpt

def run_ere(ere_data, params):
  params['current_dataset'] = 'ere'
  ere_estimator, ere_input_fn, ere_steps, ere_evaluator = prepare_ere(ere_data, params)
  if params['debug']:
    print('debug mode')
    ere_steps = 100
    params['epoch'] = 1
  for epoch in range(1, params['epoch']+1):
    print('==========')
    print('epoch', epoch)
    ere_estimator._params['current_epoch'] = epoch
    ere_estimator.train(input_fn=ere_input_fn,
                    steps=ere_steps,
                    hooks=[ere_evaluator])

  print('finish training ere, best dev (%4.4f) found at epoch: %d'
         % (ere_evaluator.best_dev, ere_evaluator.best_epoch))
  evaluate_ere_test(ere_estimator, ere_evaluator, ere_data[2], params)
  return ere_evaluator.best_ckpt

def translate_names_vocab(names, from_vocab, to_vocab):
  inv_from_vocab = {v:k for k,v in from_vocab.iteritems()}
  for i in range(len(names)):
    for j in range(len(names[i])):
      names[i][j] = to_vocab[inv_from_vocab[names[i][j]]]

def get_ere2ace05_mapping(mapping_file, ace05_label_set, ere_label_set):
  mapping = []
  with open(mapping_file, 'r') as input_file:
    for line in input_file:
      mapping += [tuple(line.split())]
  ere2ace05_index = {}

  # create index mapping
  if len(mapping[0]) == 2:
    for k, v in mapping:
      if ere_label_set[v] not in ere2ace05_index:
        ere2ace05_index[ere_label_set[v]] = []  #initialize
      ere2ace05_index[ere_label_set[v]] += [ace05_label_set[k]]
  else:
    for k, v, r in mapping:
      if ere_label_set[v] not in ere2ace05_index:
        ere2ace05_index[ere_label_set[v]] = []  #initialize
      ere2ace05_index[ere_label_set[v]] += [(ace05_label_set[k], r)]
  return ere2ace05_index   

def from_mapping_to_matrix(mapping, num_class1, num_class2, value=1.0):
  """class1 to class2 mapping"""
  matrix = np.zeros(shape=[num_class1, num_class2], dtype=np.float32)
  for class1 in mapping.keys():
    for class2 in mapping[class1]:
      if isinstance(class2, tuple):
        class2, v = class2
        matrix[class1, class2] = value
      else:
#       print(class1, class2)
        matrix[class1, class2] = value
  # matrix = np.divide(matrix,np.sum(matrix, axis=1, keepdims=True))
  return matrix

def experiment_ace05_ere(params):
  np.set_printoptions(linewidth=150)
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info(pprint.pformat(params))
  begin_time = time.time()
  ace05_data, ace05_entity_index, ace05_vocab, ace05_label, ace05_embed  = read_ace05(params)
  ere_data, ere_entity_index, ere_vocab, ere_label, ere_embed = read_ere(params)
  vocab, embed = concat_vocab(ere_vocab, ace05_vocab, ere_embed, ace05_embed)
  print('ere vocab_size', len(ere_vocab))
  print('ace05 vocab_size', len(ace05_vocab))
  print('combined vocab size', len(vocab), len(embed))
  params['vocab_size'] = len(vocab)
  params['embed'] = embed
  if params['joint'] in {'joint_kshot', 'joint_ksupport'}:
    params['kshot'] = True
  else:
    params['kshot'] = False
  ere_names, ere_names_vocab, ere_names_embed = ere.process_label_names(ere_label, vocab, embed, subtype=params['subtype'], extended_tag=False, append_tag=True)
  ace05_names, ace05_names_vocab, ace05_names_embed = ace05.process_label_names(ace05_label, vocab, embed, subtype=params['subtype'], extended_tag=False, append_tag=True, directional=params['directional'])
  names_vocab, names_embed = concat_vocab(ace05_names_vocab, ere_names_vocab, ace05_names_embed, ere_names_embed)
  print('ace05 label names', ace05_names, ace05_names_vocab)
  print('ere label names', ere_names, ere_names_vocab)
  translate_names_vocab(ere_names, ere_names_vocab, names_vocab)
  print('ere translated label names', ere_names, names_vocab)
  ere_names = utils.pad_sequences(ere_names, 15, value=0)
  ace05_names = utils.pad_sequences(ace05_names, 15, value=0)
  params['ere_relation_names'] = ere_names
  params['relation_names_embed'] = names_embed
  params['ace05_relation_names'] = ace05_names

  if params['only_train_target']:
    ere_train_Y = inputs.pick_type(ere_data[0][1], relation_types=[ere_label[params['target_type']]])
    ere_dev_Y = inputs.pick_type(ere_data[1][1], relation_types=[ere_label[params['target_type']]])
    ere_test_Y = inputs.pick_type(ere_data[2][1], relation_types=[ere_label[params['target_type']]])

    ere_data= (ere_data[0][0], ere_train_Y), (ere_data[1][0], ere_dev_Y), (ere_data[2][0], ere_test_Y)
    params['ere_num_classes'] = 2
    params['ere_relation_names'] = ere_names[[0, ere_label[params['target_type']]], :]
    print('only train target', params['ere_relation_names'])
  if params['target_percent'] < 1.0:
    ere_train = inputs.take_percentage_of_type(ere_data[0][0], ere_data[0][1], 
        relation_type=ere_label[params['target_type']],
        percentage=params['target_percent'])
    ere_data = ere_train, ere_data[1], ere_data[2]
  
  ere2ace05_index = get_ere2ace05_mapping(params['mapping_file'] if params['directional'] else params['mapping_file']+'.undirect', 
                                          ace05_label, ere_label)# only for maintype
    
  params['ere2ace05_index'] = from_mapping_to_matrix(ere2ace05_index, params['ere_num_classes'], params['ace05_num_classes'], params['similarity'])
  print('ere2ace05:', params['ere2ace05_index'])
  # convert ace05 data to be consistent with ere
  ace05_Xs = [ace05_data[0][0], ace05_data[1][0], ace05_data[2]['bc1'][0],
      ace05_data[2]['cts'][0], ace05_data[2]['wl'][0]]
  print('old', ace05_data[0][0]['word'][0], ace05_data[0][0]['type'][0])
  for X in ace05_Xs:
   translate_entity_type(X, ace05_entity_index, ere_entity_index)
   translate_vocab_index(X, ace05_vocab, vocab)

  print('new', ace05_data[0][0]['word'][0], ace05_data[0][0]['type'][0])
  if params['joint'] == 'ace':
    run_ace(ace05_data, params)
  elif params['joint'] == 'ere':
    run_ere(ere_data, params)
  elif params['joint'] == 'ace2ere':
    run_ace2ere(ace05_data, ere_data, params)
  elif params['joint'] == 'ere2ace':
    run_ere2ace(ere_data, ace05_data, params)
  elif params['joint'].startswith('joint'):
    run_joint(ace05_data, ere_data, params, ere_label_set=ere_label)

  print('total time cost: ', datetime.timedelta(seconds=time.time()-begin_time))
