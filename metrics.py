from __future__ import print_function
import os, sys
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
import tensorflow as tf
from inputs import pad_features
from utils import to_categorical
from tensorflow.python.tools import inspect_checkpoint as chkp


def f1_metric(targets, preds, nb_classes, average='micro', output_types=None):
  preds = np.argmax(preds, 1)
  targets = np.argmax(targets, 1)
  if output_types is not None:
    return f1_score(targets, preds, labels=output_types, average=average)
  else:
    return f1_score(targets, preds, labels=range(1,nb_classes), average=average)

def precision_metric(targets, preds, nb_classes):
  preds = np.argmax(preds, 1)
  targets = np.argmax(targets, 1)
  return precision_score(targets, preds, labels=range(1,nb_classes), average='micro')

def recall_metric(targets, preds, nb_classes):
  preds = np.argmax(preds, 1)
  targets = np.argmax(targets, 1)
  return recall_score(targets, preds, labels=range(1,nb_classes), average='micro')


def precision_recall(targets, preds, nb_classes):
  preds = np.argmax(preds, 1)
  targets = np.argmax(targets, 1)
  return precision_score(targets, preds, labels=range(1, nb_classes), average='micro'), recall_score(targets, preds, labels=range(1, nb_classes), average='micro')

def get_list_by_key(pred_iter, key):
  preds = []
  for pred in pred_iter:
    preds += [pred[key]]
  return preds

def evaluate_dataset(estimator, dataset, nb_classes, label_key='relation', output_types=None):
  X, Y = dataset
  input_fn = tf.estimator.inputs.numpy_input_fn(x=X, y=Y, num_epochs=1, shuffle=False)
  preds = estimator.predict(input_fn=input_fn, predict_keys=[label_key])
  preds = list(preds)
  preds = get_list_by_key(preds, label_key)
  return f1_metric(Y, preds, nb_classes, output_types)

def evaluate_predict(estimator, dataset, nb_classes, label_key='relation'):
  X, Y = dataset
  input_fn = tf.estimator.inputs.numpy_input_fn(x=X, y=Y, num_epochs=1, shuffle=False)
  preds = estimator.predict(input_fn=input_fn, predict_keys=[label_key])
  preds = list(preds)
  preds = get_list_by_key(preds, label_key)
  return f1_metric(Y, preds, nb_classes), preds



class Evaluator(tf.train.SessionRunHook):
  def __init__(self, estimator, steps_per_epoch, input_fns, Ys, nb_classes, label_key, output_types=None):
    self.estimator = estimator
    self.input_fns = input_fns
    self.Ys = Ys
    self.nb_classes = nb_classes
    self.label_key = label_key
    self.steps_per_epoch = steps_per_epoch
    self.epoch = 0
    self.best_dev = -1
    self.best_epoch = 0-0
    self.best_ckpt = self.estimator.model_dir + '/' + self.label_key +'_best/'
    self.output_types = output_types

  def evaluate_dataset(self, estimator, input_fn, Y, nb_classes, label_key='relation', average='micro', output_types=None):
    preds = estimator.predict(input_fn=input_fn, predict_keys=[label_key])
    preds = list(preds)
    preds = get_list_by_key(preds, label_key)
    return f1_metric(Y, preds, nb_classes, average=average, output_types=output_types), preds

  def evaluate_precision_recall(self, estimator, input_fn, Y, nb_classes, label_key='relation'):
    preds = estimator.predict(input_fn=input_fn, predict_keys=[label_key])
    preds = list(preds)
    preds = get_list_by_key(preds, label_key)
    f = f1_metric(Y, preds, nb_classes)
    p, r = precision_recall(Y, preds, nb_classes)
    return f, p, r

  def begin(self):
    self.best_saver = tf.train.Saver(max_to_keep=1)
    print('create dir', self.best_ckpt)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def print_tensors(self, session):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='relation_type/meta')
    np.set_printoptions(precision=2)
    for var in vars:
      print(var.name)
      var = session.run([var])
      print(var)
  
  def end(self, session, global_step_value=None):
    if global_step_value is None:
      global_step_value = session.run(tf.train.get_global_step())

    self.print_tensors(session)
    train_input_fn, dev_input_fn = self.input_fns
    train_Y, dev_Y = self.Ys
    epoch = global_step_value/int(self.steps_per_epoch)
    step = global_step_value - epoch* self.steps_per_epoch
    if epoch > self.epoch:
      train_score, _ = self.evaluate_dataset(self.estimator, train_input_fn, train_Y, self.nb_classes, self.label_key, output_types=self.output_types)
    else:
      train_score = None # only eval train every epoch
    dev_score, _ = self.evaluate_dataset(self.estimator, dev_input_fn, dev_Y, self.nb_classes, self.label_key, output_types=self.output_types)
    if train_score is None:
      print(self.label_key + ' at epoch: %s and step: %d, dev (%4.2f)' %
          (epoch, step, 100.0*dev_score))
    else:
      print(self.label_key + ' at epoch: %s and step: %d, train (%4.2f), dev (%4.2f)' % (epoch, step, 100.0*train_score, 100.0*dev_score))
    if dev_score > self.best_dev:
      self.best_dev = dev_score
      self.best_epoch = str(epoch) + '-' + str(step) 
      path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(epoch),
                            global_step_value)
      print('new best model saved to', path)
    self.epoch = epoch
    return train_score, dev_score

class EvaluatorACE05EREHook(tf.train.SessionRunHook):
  def __init__(self, estimator, steps_per_epoch,
               ace05_train_eval_fn, ace_dev_input_fn, ace05_train_Y, ace05_dev_Y, ace05_nb_classes,
               ere_train_eval_fn, ere_dev_input_fn, ere_train_Y, ere_dev_Y, ere_nb_classes, output_types=None):
    self.ace05_evaluator = Evaluator(estimator, steps_per_epoch, (ace05_train_eval_fn, ace_dev_input_fn),
        (ace05_train_Y, ace05_dev_Y), ace05_nb_classes, 'ace05_relation')
    self.ere_evaluator = Evaluator(estimator, steps_per_epoch, (ere_train_eval_fn, ere_dev_input_fn), 
        (ere_train_Y, ere_dev_Y), ere_nb_classes, 'ere_relation', output_types=output_types)
  
  def begin(self):
    self.ace05_evaluator.begin()
    self.ere_evaluator.begin()

  def end(self, session):
    train_score, dev_score = self.ace05_evaluator.end(session)
    #tf.summary.scalar('ace05_train_score', train_score)
    #tf.summary.scalar('ace05_dev_score', dev_score)
    train_score, dev_score = self.ere_evaluator.end(session)
    #tf.summary.scalar('ere_train_score', train_score)
    #tf.summary.scalar('ere_dev_score', dev_score)

class EvaluatorACE05EREListener(tf.train.CheckpointSaverListener):
  def __init__(self, estimator, steps_per_epoch, ace05_train_eval_fn, ace_dev_input_fn, ace05_train_Y, ace05_dev_Y, ace05_nb_classes,
               ere_train_eval_fn, ere_dev_input_fn, ere_train_Y, ere_dev_Y, ere_nb_classes):
    self.ace05_evaluator = Evaluator(estimator, steps_per_epoch, (ace05_train_eval_fn, ace_dev_input_fn),
        (ace05_train_Y, ace05_dev_Y), ace05_nb_classes, 'ace05_relation')
    self.ere_evaluator = Evaluator(estimator, steps_per_epoch, (ere_train_eval_fn, ere_dev_input_fn),
        (ere_train_Y, ere_dev_Y), ere_nb_classes, 'ere_relation')

  def begin(self):
    self.ace05_evaluator.begin()
    self.ere_evaluator.begin()

  def after_save(self, session, global_step_value):
    self.ace05_evaluator.end(session, global_step_value)
    self.ere_evaluator.end(session, global_step_value)

class EvaluatorACE05Hook(tf.train.SessionRunHook):
  def __init__(self, estimator, ace05data):
    self.estimator = estimator
    self.ace05data = ace05data
    self.epoch = 1
    self.best_dev = -1
    self.best_epoch = 0
    self.best_ckpt = self.estimator.model_dir+'/best/'
    

  def begin(self):
    self.best_saver = tf.train.Saver(max_to_keep=1)
    print('create dir', self.best_ckpt)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def end(self, session):
    global_step_value = tf.train.get_global_step()
    train, dev, test = self.ace05data
    train_score = evaluate_dataset(self.estimator, train, 11)
    dev_score = evaluate_dataset(self.estimator, dev, 11)
    print('At epoch: %d, train (%4.2f), dev (%4.2f)' % 
        (self.epoch, 100.0*train_score.item(), 100.0*dev_score.item()))
    if dev_score > self.best_dev:
      self.best_dev = dev_score
      self.best_epoch = self.epoch
      path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(self.epoch), 
                            global_step_value)
      print('new best model saved to', path)
    self.epoch += 1


class EvaluatorNYT10Unified(tf.train.SessionRunHook):
  def __init__(self, estimator, name, save_best, data, batch_size):
    self.estimator = estimator
    self.name = name
    self.save_best = save_best
    self.data = data
    self.batch_size = batch_size
    self.epoch = 1
    self.best_dev = -1
    self.best_epoch = 0
    self.best_ckpt = self.estimator.model_dir+'/best/'
    self.nb_classes = 53

  def begin(self):
    if not self.save_best:
      return
    self.best_saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def get_prediction(self, X):
    input_fn = tf.estimator.inputs.numpy_input_fn(x=X, y=None,
      batch_size=self.batch_size, num_epochs=1, shuffle=False)
    preds = self.estimator.predict(input_fn=input_fn, predict_keys=['relation'])
    preds = list(preds)
    preds = get_list_by_key(preds, 'relation')
    return preds


  def get_pr_curve(self, X, Y):
    preds = self.get_prediction(X)
    preds = np.array(preds)
    Y = Y[X['mask']]
    precision = {}
    recall = {}
    # for i in range(self.nb_classes):
      # precision[i], recall[i], _ = precision_recall_curve(Y[:, i], preds[:, i])
    #micro average not counting 0 which is NA

    f1_score = f1_metric(Y, preds, self.nb_classes)
    precision_score = precision_metric(Y, preds, self.nb_classes)
    recall_score = recall_metric(Y, preds, self.nb_classes)
    print('F1', '%4.4f' % f1_score, 'P', '%4.4f' % precision_score, 'R', '%4.4f' % recall_score)
    precision["micro"], recall["micro"], _ = precision_recall_curve(
      Y[:, 1:].ravel(), preds[:, 1:].ravel())

    average_precision = average_precision_score(
      Y[:, 1:], preds[:, 1:], average='micro')
    return precision, recall, average_precision

  def end(self, session):
    global_step_value = tf.train.get_global_step()
    # print('all')
    # dev_score = self.evaluate_all()
    p, r, dev_score = self.get_pr_curve(self.data[0], self.data[1])
    print(self.name, 'average_precision_score', dev_score)
    if self.save_best and dev_score > self.best_dev:
      self.best_dev = dev_score
      self.best_epoch = str(self.epoch)
      path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(self.epoch), 
                            global_step_value)
      print('new best model saved to', path)
    self.epoch += 1

class EvaluatorNYT10UnifiedListener(tf.train.CheckpointSaverListener):
  def __init__(self, estimator, name, save_best, data, batch_size):
    self.estimator = estimator
    self.name = name
    self.save_best = save_best
    self.data = data
    self.batch_size = batch_size
    self.epoch = 1
    self.best_dev = -1
    self.best_epoch = 0
    self.best_ckpt = self.estimator.model_dir+'/best/'
    self.nb_classes = 53

  def begin(self):
    if not self.save_best:
      return
    self.best_saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def get_prediction(self, X):
    input_fn = tf.estimator.inputs.numpy_input_fn(x=X, y=None,
      batch_size=self.batch_size, num_epochs=1, shuffle=False)
    preds = self.estimator.predict(input_fn=input_fn, predict_keys=['relation'])
    preds = list(preds)
    preds = get_list_by_key(preds, 'relation')
    return preds

  def get_pr_curve(self, X, Y):
    preds = self.get_prediction(X)
    preds = np.array(preds)
    Y = Y[X['mask']]
    precision = {}
    recall = {}
    # for i in range(self.nb_classes):
      # precision[i], recall[i], _ = precision_recall_curve(Y[:, i], preds[:, i])
    #micro average not counting 0 which is NA

    f1_score = f1_metric(Y, preds, self.nb_classes)
    precision_score = precision_metric(Y, preds, self.nb_classes)
    recall_score = recall_metric(Y, preds, self.nb_classes)
    print('F1', '%4.4f' % f1_score, 'P', '%4.4f' % precision_score, 'R', '%4.4f' % recall_score)
    precision["micro"], recall["micro"], _ = precision_recall_curve(
      Y[:, 1:].ravel(), preds[:, 1:].ravel())

    #average_precision = auc(precision['micro'], recall['micro'])
    average_precision = average_precision_score(
        Y[:, 1:], preds[:, 1:], average='micro')

    return precision, recall, average_precision

  def after_save(self, session, global_step_value):
    # print('all')
    # dev_score = self.evaluate_all()
    p, r, dev_score, top_100, top_200, top_300 = self.get_pr_curve(self.data[0], self.data[1])
    print('epoch', self.epoch, self.name, 'average_precision_score', dev_score)
    if self.save_best and dev_score > self.best_dev:
      self.best_dev = dev_score
      self.best_epoch = str(self.epoch)
      path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(self.epoch), 
                            global_step_value)
      print('new best model saved to', path)
    sys.stdout.flush()
    self.epoch += 1


class EvaluatorNYT10MIMLListener(EvaluatorNYT10UnifiedListener):
  def __init__(self, estimator, name, save_best, data, batch_size):
        super(EvaluatorNYT10MIMLListener, self).__init__(estimator, name, save_best, data, batch_size)

  def f1_metric(self, targets, preds, nb_classes):
    preds = np.round(preds)
    return f1_score(targets[:, 1:], preds[:, 1:], average='micro')
  
  def get_pr_curve(self, X, Y):
    preds = self.get_prediction(X)
    preds = np.array(preds)
    Y = Y[X['mask']]
    precision = {}
    recall = {}
    # for i in range(self.nb_classes):
      # precision[i], recall[i], _ = precision_recall_curve(Y[:, i], preds[:, i])
    #micro average not counting 0 which is NA
    f1_score = self.f1_metric(Y, preds, self.nb_classes)
    print('F1', '%4.4f' % f1_score)
    precision["micro"], recall["micro"], _ = precision_recall_curve(
      Y[:, 1:].ravel(), preds[:, 1:].ravel())
    average_precision = auc(recall['micro'], precision['micro'])
    top_100_p = self.top_n_precision(Y, preds, 100)
    top_200_p = self.top_n_precision(Y, preds, 200)
    top_300_p = self.top_n_precision(Y, preds, 300)
    #average_precision = average_precision_score(
    #    Y[:, 1:], preds[:, 1:], average='micro')
    return precision, recall, average_precision, top_100_p, top_200_p, top_300_p
  
  def top_n_precision(self, Y, preds, n):
    y = Y[:, 1:].ravel()
    h = preds[:, 1:].ravel()
    desc_index = np.argsort(h)[::-1]

    y = y[desc_index]
    h = h[desc_index]
    h = np.round(h)
    p = precision_score(y[:n], h[:n], pos_label=1.0)
    print(p)
    return p

class EvaluatorNYT10Joint(tf.train.SessionRunHook):
  def __init__(self, estimator, input_fn, data, name, save_best):
    self.estimator = estimator
    self.name = name
    self.save_best = save_best
    self.input_fn = input_fn
    self.data = data
    self.epoch = 1
    self.best_dev = -1
    self.best_epoch = 0
    self.best_ckpt = self.estimator.model_dir+ '/' + name + '_best/'
    self.nb_classes = 53

  def begin(self):
    if not self.save_best:
      return
    self.best_saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def get_prediction(self, input_fn, predict_key='nyt10_relation'):
    preds = self.estimator.predict(input_fn=input_fn, predict_keys=[predict_key])
    preds = list(preds)
    preds = get_list_by_key(preds, predict_key)
    return preds

  def f1_metric(self, targets, preds, nb_classes):
    preds = np.round(preds)
    return f1_score(targets[:, 1:], preds[:, 1:], average='micro')

  def get_pr_curve(self, input_fn, data):
    X, Y = data
    preds = self.get_prediction(input_fn)
    preds = np.array(preds)
    Y = Y[X['mask']]
    precision = {}
    recall = {}
    # for i in range(self.nb_classes):
      # precision[i], recall[i], _ = precision_recall_curve(Y[:, i], preds[:, i])
    #micro average not counting 0 which is NA

    f1_score = self.f1_metric(Y, preds, self.nb_classes)
    print('F1', f1_score)
    precision["micro"], recall["micro"], _ = precision_recall_curve(
      Y[:, 1:].ravel(), preds[:, 1:].ravel())

    #average_precision = average_precision_score(
    #    Y[:, 1:], preds[:, 1:], average='micro')
    average_precision = auc(recall['micro'], precision['micro'])
    top_100_p = self.top_n_precision(Y, preds, 100)
    top_200_p = self.top_n_precision(Y, preds, 200)
    top_300_p = self.top_n_precision(Y, preds, 300)
    
    return precision, recall, average_precision, top_100_p, top_200_p, top_300_p

  def top_n_precision(self, Y, preds, n):
    y = Y[:, 1:].ravel()
    h = preds[:, 1:].ravel()
    desc_index = np.argsort(h)[::-1]

    y = y[desc_index]
    h = h[desc_index]
    h = np.round(h)
    p = precision_score(y[:n], h[:n], pos_label=1.0)
    print(p)
    return p

  def print_tensors(self, session):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='relation_type/meta')
    np.set_printoptions(precision=2)
    for var in vars:
      print(var.name)
      var = session.run([var])
      print(var)

  def end(self, session):
    self.print_tensors(session)
    global_step_value = tf.train.get_global_step()
    # print('all')
    # dev_score = self.evaluate_all()
    p, r, dev_score,_,_,_ = self.get_pr_curve(self.input_fn, self.data)
    print(self.name, 'average_precision_score', dev_score)
    if self.save_best and dev_score > self.best_dev:
      self.best_dev = dev_score
      self.best_epoch = str(self.epoch)
      path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(self.epoch), 
                            global_step_value)
      print('new best model saved to', path)
    self.epoch += 1

class EvaluatorACE05Test(tf.train.SessionRunHook):
  def __init__(self, estimator, input_fn, test, label_key, num_classes, name="evaluater"):
    self.evaluator = {}
    for corpus in input_fn:
      self.evaluator[corpus] = EvaluatorLight(estimator, input_fn[corpus], test[corpus][1], label_key, num_classes, False, name + '_' + corpus)

  def end(self, session):
    for e in self.evaluator.values():
      e.end(session)

  def evaluate_all(self):
    scores = {}
    for corpus, e in self.evaluator.iteritems():
      score = e.evaluate_dataset(e.estimator, e.input_fn, e.Y)
      scores[corpus] = score
    return scores
    
class EvaluatorLight(tf.train.SessionRunHook):
  def __init__(self, estimator, input_fn, Y, label_key, num_classes, save_best=False, name="evaluater"):
    self.estimator = estimator
    self.Y = Y
    self.input_fn = input_fn
    self.num_classes = num_classes
    self.label_key = label_key
    self.save_best = save_best
    self.name = name
    self.epoch = 1
    self.best_dev = -1
    self.best_epoch = 0
    self.best_ckpt = self.estimator.model_dir + '/' + name +'_best/'

  def begin(self):
    if not self.save_best:
      return
    self.best_saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def evaluate_dataset(self, estimator, input_fn, Y):
    preds = estimator.predict(input_fn=input_fn, predict_keys=[self.label_key])
    preds = list(preds)
    preds = get_list_by_key(preds, self.label_key)
    return f1_metric(Y, preds, self.num_classes)
 
  def end(self, session):
    global_step_value = tf.train.get_global_step()
    score = self.evaluate_dataset(self.estimator, self.input_fn, self.Y)
    print('At epoch: %d, %s %4.2f'  %
        (self.epoch, self.name, 100.0*score))
    if score > self.best_dev:
      self.best_dev = score
      self.best_epoch = self.epoch
      if self.save_best:
        path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(self.epoch),
                            global_step_value)
        print('new best model saved to', path)
    self.epoch += 1
    
class EvaluatorEREHook(tf.train.SessionRunHook):
  def __init__(self, estimator, ere_data):
    self.estimator = estimator
    self.ere_data = ere_data
    self.epoch = 1
    self.best_dev = -1
    self.best_epoch = 0
    self.best_ckpt = self.estimator.model_dir+'/best/'

  def begin(self):
    self.best_saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists(self.best_ckpt):
      os.mkdir(self.best_ckpt)

  def end(self, session):
    global_step_value = tf.train.get_global_step()
    train, dev, test = self.ere_data
    train_score = evaluate_dataset(self.estimator, train, 6)
    dev_score = evaluate_dataset(self.estimator, dev, 6)
    print('At epoch: %d, train (%4.2f), dev (%4.2f)' % 
        (self.epoch, 100.0*train_score.item(), 100.0*dev_score.item()))
    if dev_score > self.best_dev:
      self.best_dev = dev_score
      self.best_epoch = self.epoch
      path = self.best_saver.save(session, self.best_ckpt+'epoch-'+str(self.epoch), 
                            global_step_value)
      print('new best model saved to', path)
    self.epoch += 1
