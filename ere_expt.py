from __future__ import print_function

import tensorflow as tf
import random

import ere
from utils import to_categorical
from inputs import pad_features, take_percentage
import metrics
from models import relation_model, relation_stack_model


def preprocess(data, params):
  # change list of dictionary to dictionary of lists
  data = dict(zip(data[0],zip(*[d.values() for d in data])))
  Y = data.pop('y')
  X = data
  X = pad_features(X, params)
  Y = to_categorical(Y, nb_classes=params['num_classes'])
  return X, Y

def experiment_ere(params):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info(params)
  train, dev, test, vocab, label_set, word_embed = ere.load_dataset(
      vocab_size=params['vocab_size'], max_len=params['max_len'])
  params['embed'] = word_embed
  random.shuffle(train)
  trainX, trainY = preprocess(train, params)
  print(trainX)
  print(trainY)
  devX, devY = preprocess(dev, params)
  testX, testY = preprocess(test, params)


  trainX, trainY = take_percentage(trainX, trainY, params['percent_train'])
  
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x=trainX, y=trainY, 
      num_epochs=1, batch_size=params['batch_size'], shuffle=True)
  num_steps_per_epoch = trainY.shape[0]/params['batch_size'] + 1
  params['learning_rate_decay_step'] = num_steps_per_epoch*params['lr_decay_epoch']

  if params['stack']:
    model_fn = relation_stack_model
  else:
    model_fn = relation_model

  config = tf.estimator.RunConfig()
  config = config.replace(tf_random_seed=params['random_seed'], keep_checkpoint_max=1)
  estimator = tf.estimator.Estimator(model_dir=params['model_dir'], model_fn=model_fn, 
                                     params=params,
                                     config=config)

  print('num steps per epoch', num_steps_per_epoch)
  print('start training')
  evaluator = metrics.EvaluatorEREHook(
      estimator, ((trainX, trainY), (devX, devY), (testX, testY)))

  if params['debug']:
    print('debug mode')
    num_steps_per_epoch = 100
    params['epoch'] = 1
  for epoch in range(1, params['epoch']+1):
    print('==========')
    print('epoch', epoch)
    estimator.train(input_fn=train_input_fn,
                    steps=num_steps_per_epoch,
                    hooks=[evaluator])

  print('finish training, best dev (%4.4f) found at epoch: %d'
         % (evaluator.best_dev, evaluator.best_epoch))

  estimator._model_dir = evaluator.best_ckpt
  test_score = metrics.evaluate_dataset(estimator, (testX, testY), params['num_classes'])
  with open(params['score_file'],'a') as score_output:
    print('dev % 4.2f' % (100.0*evaluator.best_dev.item()), end=' ')
    print('dev % 4.2f' % (100.0*evaluator.best_dev.item()), end=' ', file=score_output)
    print('test % 4.2f' % (100.0*test_score.item()), end=' ')
    print('test % 4.2f' % (100.0*test_score.item()), end=' ', file=score_output)
    print('epoch', evaluator.best_epoch)
    print('epoch', evaluator.best_epoch, file=score_output)
