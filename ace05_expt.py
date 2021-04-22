from __future__ import print_function

import numpy as np
import tensorflow as tf
import ace05
from utils import to_categorical

from inputs import pad_features, take_percentage
import metrics
from models import relation_model, relation_stack_model


def preprocess_ace05(datasets, params):
  train, dev, test = datasets
  train = preprocess(train, params)
  dev = preprocess(dev, params)
  processed_test = {}
  for corpus in test:
    processed_test[corpus] = preprocess(test[corpus], params)
  return train, dev, processed_test

def preprocess(dataset, params):
  X, Y = dataset
  X = pad_features(X, params)
  Y = to_categorical(Y, nb_classes=params['num_classes'])
  return X, Y

def experiment_ace05(params):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info(params)
  ace05_data, embed = ace05.load_dataset(max_len=params['max_len'])
  ace05_data = preprocess_ace05(ace05_data, params)
  train, dev, test = ace05_data
  params['embed'] = embed
  trainX, trainY = train
  devX, devY = dev

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
  config = config.replace(tf_random_seed=params['random_seed'], keep_checkpoint_max=1,)
  estimator = tf.estimator.Estimator(model_dir=params['model_dir'], model_fn=model_fn, 
                                     params=params,
                                     config=config)

  print('num steps per epoch', num_steps_per_epoch)
  print('start training')

  evaluator = metrics.EvaluatorACE05Hook(estimator, ace05_data)

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
  with open(params['score_file'],'a') as score_output:
    estimator._model_dir = evaluator.best_ckpt
    print('bc0 %4.2f' % (100.0*evaluator.best_dev.item()), end=' ')
    print('bc0 %4.2f' % (100.0*evaluator.best_dev.item()), end=' ', file=score_output)
    all_preds, all_truth = [], []
    for corpus in test:
      score, preds = metrics.evaluate_predict(estimator, test[corpus], params['num_classes'])
      all_preds += [preds]
      all_truth += [test[corpus][1]]
      print(corpus,'%4.2f' % (100.0*score.item()), end=' ')
      print(corpus,'%4.2f' % (100.0*score.item()), end=' ', file=score_output)
    all_preds = np.concatenate(all_preds, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)
    micro_avg = metrics.f1_metric(all_truth, all_preds, params['num_classes'])
    print('micro', '%4.2f' % (100.0*micro_avg), end=' ' )
    print('micro', '%4.2f' % (100.0*micro_avg), end=' ', file=score_output)
    print('epoch', evaluator.best_epoch)
    print('epoch', evaluator.best_epoch, file=score_output)

