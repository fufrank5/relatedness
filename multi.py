from __future__ import print_function
import os
import sys
import tempfile
import tensorflow as tf

from ace05_expt import experiment_ace05
from ere_expt import experiment_ere
from ace05_ere_expt import experiment_ace05_ere

flags = tf.app.flags
flags.DEFINE_string('expt', 'ace05_ere', 'which dataset to run: ace05, nyt10, ere')
flags.DEFINE_string('model', 'bi_rnn_att', 'which model to run: cnn, rnn, bi_rnn')
flags.DEFINE_boolean('stack', False, 'use stack model or not')
flags.DEFINE_integer('random_seed', 42, 'random seed for tensorflow init')
flags.DEFINE_string('data_path', './data/ace05/', 
                   'data path for preprocessed features relation.pkl')
flags.DEFINE_string('score_path', './scores/', 'file to save the scores')
flags.DEFINE_string('pred_path', './preds/', 'file to save the predictions')
flags.DEFINE_bool('predict', False, 'whether to store the output in file')
flags.DEFINE_string('predict_model_dir', './models/', 'model dir for prediction')
flags.DEFINE_string('model_dir', './models/', 'top model dir for training')
flags.DEFINE_bool('pretrain_w2v', True, 'use pretrained word embedding')
flags.DEFINE_integer('epoch', 10, 'num of epoches to train')
flags.DEFINE_bool('keep_score', True, 'whethter to write scores in file')
flags.DEFINE_bool('extra_feature', False, 'add extra features such as dep, chunk')
flags.DEFINE_bool('lr_decay', False, 'learning rate decay function')
flags.DEFINE_integer('max_len', 100, 'max length for sentence')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('lr_decay_epoch', 3, 'epochs to half learning rate, 1 for ere, 3 for joint learning')
flags.DEFINE_float('lr_decay_rate', 0.5, 'lr_decay_rate')
flags.DEFINE_bool('lr_staircase', True, 'lr_staircase')
flags.DEFINE_float('percent_train', 1.0,
                     'percentage of labeled data for training 0.0-1.0')
flags.DEFINE_float('ace05_percent_train', 1.0, 'for joint learning')
flags.DEFINE_float('ere_percent_train', 1.0, 'for joint learning')
flags.DEFINE_float('nyt10_percent_train', 1.0, 'for joint learning')
flags.DEFINE_integer('ace05_kshot', 0, 'kshot size')
flags.DEFINE_integer('ere_kshot', 0, 'kshot size')
flags.DEFINE_float('pathnre_percent_train', 1.0, 'for joint learning')
flags.DEFINE_integer('pos_embed_dim', 50, 'dimension for position embeddding')
flags.DEFINE_integer('num_hidden_layer', 1, 'number of hidden layers')
flags.DEFINE_integer('hidden_layer_dim', 300, 'dimension of the hidden layer')
flags.DEFINE_integer('rnn_state_size', 150, 'rnn state size')
flags.DEFINE_integer('hidden_rnn_size', 150, 'rnn state size for the lower level in the stack model')
flags.DEFINE_integer('num_filters', 150, 'num of filters')
flags.DEFINE_string('kernel_size', '2 3 4 5', 'cnn window/kernel size')
flags.DEFINE_integer('bag_size', 64, 'relation bag size for nyt10')
flags.DEFINE_bool('tune_word_embed', False, 'tune word embed or not')
flags.DEFINE_string('tag','','additional tag to score_file, output_file')
flags.DEFINE_string('joint','joint_ksupport_model','how to do joint learning')
flags.DEFINE_bool('use_existing_pretrain', False, 
    'load encoder part of pretrained model. This switch is only to'
    ' decide whether to load an existing model or train a model')
flags.DEFINE_bool('debug', False, 'whether debug or not')
flags.DEFINE_string('train_mode', 'all', 'train part of the net,'
    ' e.g. decoder: train decoder only')
flags.DEFINE_float('lamda', 0.8, 'lamda should be between 0.0 to 1.0,'
    'loss=(1-lamda)*loss_ace05+lamda*loss_ere')
flags.DEFINE_float('alpha3', 0.01, 'adversarial loss, loss = relation_loss + alpha3 * domain_loss')
flags.DEFINE_float('alpha_meta', 0.01, 'meta classifier loss for is_relation')
flags.DEFINE_bool('meta_classifier', False, 'whether to use meta classifier')
flags.DEFINE_string('meta_input', 'feature', 'feature, hidden, output')
flags.DEFINE_bool('da', False, 'whether to use domain adversarial')
flags.DEFINE_bool('l2', False, 'whether to use l2 loss for regularization (on repr'
                               'instead of parameters)')
flags.DEFINE_string('da_input', 'feature', 'feature, hidden')
flags.DEFINE_integer('word_context_dim', 100, 'word level attention vector dimension')
flags.DEFINE_string('joint_decoder', 'backprop', 'forward, backprop and cascade')
flags.DEFINE_bool('encoder_hidden_layer', False, 'add one shared hidden layer after encoder')
flags.DEFINE_bool('shared_hidden', False, 'shared hidden layer for decoders')
flags.DEFINE_integer('shared_hidden_dim', 200, 'dimension of the shared hidden layer')
flags.DEFINE_float('eval_every_n_epoch', 1.0, 'how frequent to eval in terms of epoch')

flags.DEFINE_string('mil_reduce', 'max', 'max, attention')
flags.DEFINE_integer('sent_context_dim', 100, 'sentence level attention for selecting sentence')
flags.DEFINE_bool('reduce_at_output', True,'reduce at output layer or feature layer')
flags.DEFINE_string('extra_info', 'None', 'what info to put into attention')
flags.DEFINE_bool('no_dev', 'False', 'no dev train split, keep original data split with only trian and test')
flags.DEFINE_float('neg_weight', 1.0, 'weight in loss for negatives')
flags.DEFINE_float('input_dropout', 0.0, 'dropout on word embedding')
flags.DEFINE_bool('miml', True, 'miml or not')
flags.DEFINE_string('encoder', 'bi_rnn_output', 'stack, bi_rnn_output, hidden_bi_rnn for nyt10 model, stack_num for stack')
flags.DEFINE_integer('stack_num', 1, 'default 1 encoder')
flags.DEFINE_string('decoder', 'miml','mil, miml, type_reader, type_reader_att')
flags.DEFINE_bool('binary_aux', False,'use binary labels (relation vs non-relation) for auxiliary relation task')
flags.DEFINE_string('nyt10_embed', 'nyt10', 'nyt10, google, pathnre')
flags.DEFINE_integer('relation_embed_size', 100, 'input relation embed size according to the relation definition')
flags.DEFINE_float('label_smooth', 0.0, 'label smoothing at loss function')
flags.DEFINE_string('relation_type_reader', 'random', 'mean, random ,rnn')
flags.DEFINE_bool('subtype', False, 'use type or subtype for labels')
flags.DEFINE_string('r_repr', None, 'None or extra')
flags.DEFINE_bool('opt_other', True, 'optimize other class or not')
flags.DEFINE_float('beta', 0.0, 'similarity loss in joint experiments')
flags.DEFINE_float('constraint_loss', 0.0, 'constraint for r_repr')
flags.DEFINE_bool('normalize_constraint', True, 'normalize r_repr vector for cosine distance')
flags.DEFINE_bool('output_r_repr', False, 'print r_repr in tsv file')
flags.DEFINE_string('similarity_reduce', 'avg', 'avg and hier_avg')
flags.DEFINE_float('similarity_threshold', 0.0, 'similartiy loss theshold for cosine value > th')
flags.DEFINE_bool('similarity_constraint', False, 'constraint loss for different types mapped to same type')
flags.DEFINE_string('target_type', '', 'target relation type (to reduce training data)')
flags.DEFINE_float('target_percent', 1.0, 'reduce target relation type training instances to this percentage')
flags.DEFINE_bool('only_train_target', False, 'only train target type')
flags.DEFINE_string('ere_split','0.8', '[0.8, 0.1, 0.1]')
flags.DEFINE_string('mapping_file', './data/ace05/ace2ere_1.0.txt', 'mapping file for the type correlation')
flags.DEFINE_string('embedding_dist', 'l1', 'cosine or l1 for distance metric of label embedding')
flags.DEFINE_float('similarity', 1.0, 'similarity for two relation type embedding')
flags.DEFINE_float('augment', 0.0, 'use pseudo labels for each other')
flags.DEFINE_float('augment_ace05', 0.0, 'use pseudo labels for each other')
flags.DEFINE_float('augment_nyt10', 0.0, 'use pseudo labels for each other')
flags.DEFINE_float('augment_ere', 0.0, 'use pseudo labels for each other')
flags.DEFINE_string('augment_soft_label','direct', 'direct, positvie, norm, softmax')
flags.DEFINE_bool('directional', True, 'if False  use undirectional label set for ACE05')
flags.DEFINE_integer('ksupport_size', 50, 'size for each type')
flags.DEFINE_string('kshot_combine', 'att', 'use attention score to combine the k shot examples')
flags.DEFINE_bool('dynamic_lamda', True, 'lamda adjust to number of examples (batch size) in different datasets or not')
flags.DEFINE_string('pretrained_model',"", 'load from the path if not empty')
flags.DEFINE_float('rank_pos_margin', 2.5, 'margin for positvies in the ranking loss')
flags.DEFINE_float('rank_neg_margin', 0.5, 'margin for negatives in the ranking loss')
flags.DEFINE_float('rank_scale', 2, 'rho for scaling in ranking loss computation')
FLAGS = flags.FLAGS

DATA_PATH = FLAGS.data_path

params_ace05 = {
  'num_classes': 11,
  'vocab_size': 14811, #including special tokens <s> <UNK> </s>
  'entity_type_size': 9, # including Other
  'model': FLAGS.model,
  'stack': FLAGS.stack,
  'pretrain_w2v': FLAGS.pretrain_w2v,
  'word_embed_dim': 300,
  'pos_embed_dim': FLAGS.pos_embed_dim,
  'hidden_layer_dim': FLAGS.hidden_layer_dim,
  'embed': None,
  'num_filters': FLAGS.num_filters,
  'hidden_rnn_size': FLAGS.hidden_rnn_size,
  'rnn_state_size': FLAGS.rnn_state_size,
  'max_len': FLAGS.max_len,
  'batch_size': FLAGS.batch_size,
  'epoch': FLAGS.epoch,
  'optimizer': 'adam',
  'learning_rate': FLAGS.learning_rate,
  'lr_decay': FLAGS.lr_decay,
  'lr_decay_epoch': FLAGS.lr_decay_epoch,
  'lr_decay_rate': FLAGS.lr_decay_rate,
  'lr_staircase': FLAGS.lr_staircase,
  'score_path': FLAGS.score_path,
  'model_dir': FLAGS.model_dir,
  'dep_size': 46,
  'dep_path_len': 5,
  'extra_feature': FLAGS.extra_feature,
  'chunk_type_size': 24,
  'percent_train': FLAGS.percent_train,
  'random_seed': FLAGS.random_seed,
  'tune_word_embed': FLAGS.tune_word_embed,
  'word_context_dim': FLAGS.word_context_dim,
  'debug': FLAGS.debug,
  'input_dropout': FLAGS.input_dropout,
  }

params_ere = {
  'num_classes': 6,
  'vocab_size': 20000,  # 25227 in total
  'entity_type_size': 9,
  'model': FLAGS.model,
  'stack': FLAGS.stack,
  'pretrain_w2v': FLAGS.pretrain_w2v,
  'word_embed_dim': 300,
  'pos_embed_dim': FLAGS.pos_embed_dim,
  'hidden_layer_dim': FLAGS.hidden_layer_dim,
  'embed': None,
  'num_filters': FLAGS.num_filters,
  'hidden_rnn_size': FLAGS.hidden_rnn_size,
  'rnn_state_size': FLAGS.rnn_state_size,
  'max_len': FLAGS.max_len,
  'batch_size': FLAGS.batch_size,
  'epoch': FLAGS.epoch,
  'optimizer': 'adam',
  'learning_rate': FLAGS.learning_rate,
  'lr_decay': FLAGS.lr_decay,
  'lr_decay_epoch': FLAGS.lr_decay_epoch,
  'lr_decay_rate': FLAGS.lr_decay_rate,
  'lr_staircase': FLAGS.lr_staircase,
  'score_path': FLAGS.score_path,
  'model_dir': FLAGS.model_dir,
  'extra_feature': FLAGS.extra_feature,
  'percent_train': FLAGS.percent_train,
  'random_seed': FLAGS.random_seed,
  'tune_word_embed': FLAGS.tune_word_embed,
  'word_context_dim': FLAGS.word_context_dim,
  'debug': FLAGS.debug,
  'input_dropout': FLAGS.input_dropout,
}


params_ace05_ere = {
  'joint': FLAGS.joint,
  'joint_decoder': FLAGS.joint_decoder,
  'dynamic_lamda': FLAGS.dynamic_lamda,
  'lamda': FLAGS.lamda, 
  'ere_num_classes': 9,
  'directional': FLAGS.directional,
  'ace05_num_classes': 11,
  'ere_vocab_size': 20000,  # 25227 in total
  'ace05_vocab_size': 14811,
  'entity_type_size': 9,
  'model': FLAGS.model,
  'pretrain_w2v': FLAGS.pretrain_w2v,
  'word_embed_dim': 300,
  'dep_size': 46,
  'dep_path_len': 5,
  'extra_feature': FLAGS.extra_feature,
  'chunk_type_size': 24,
  'pos_embed_dim': FLAGS.pos_embed_dim,
  'hidden_layer_dim': FLAGS.hidden_layer_dim,
  'num_hidden_layer': FLAGS.num_hidden_layer,
  'embed': None,
  'pretrained_model': FLAGS.pretrained_model,
  'use_existing_pretrain': FLAGS.use_existing_pretrain,
  'num_filters': FLAGS.num_filters,
  'rnn_state_size': FLAGS.rnn_state_size,
  'hidden_rnn_size': FLAGS.hidden_rnn_size,
  'max_len': FLAGS.max_len,
  'batch_size': FLAGS.batch_size,
  'epoch': FLAGS.epoch,
  'optimizer': 'adam',
  'learning_rate': FLAGS.learning_rate,
  'lr_decay': FLAGS.lr_decay,
  'lr_decay_epoch': FLAGS.lr_decay_epoch,
  'lr_decay_rate': FLAGS.lr_decay_rate,
  'lr_staircase': FLAGS.lr_staircase,
  'score_path': FLAGS.score_path,
  'model_dir': FLAGS.model_dir,
  'extra_feature': FLAGS.extra_feature,
  'percent_train': FLAGS.percent_train,
  'ace05_percent_train': FLAGS.ace05_percent_train,
  'ere_percent_train': FLAGS.ere_percent_train,
  'random_seed': FLAGS.random_seed,
  'tune_word_embed': FLAGS.tune_word_embed,
  'word_context_dim': FLAGS.word_context_dim,
  'train_mode': FLAGS.train_mode,
  'da': FLAGS.da,
  'l2': FLAGS.l2,
  'da_input': FLAGS.da_input,
  'alpha3': FLAGS.alpha3,
  'meta_classifier': FLAGS.meta_classifier,
  'meta_input': FLAGS.meta_input,
  'alpha_meta': FLAGS.alpha_meta,
  'encoder_hidden_layer': FLAGS.encoder_hidden_layer,
  'shared_hidden': FLAGS.shared_hidden,
  'shared_hidden_dim': FLAGS.shared_hidden_dim,
  'predict': FLAGS.predict,
  'predict_model_dir': FLAGS.predict_model_dir,
  'eval_every_n_epoch': FLAGS.eval_every_n_epoch,
  'debug': FLAGS.debug,
  'relation_embed_size': FLAGS.relation_embed_size,
  'relation_type_reader': FLAGS.relation_type_reader,
  'subtype': FLAGS.subtype,
  'r_repr': FLAGS.r_repr,
  'opt_other': FLAGS.opt_other,
  'beta': FLAGS.beta,
  'constraint_loss': FLAGS.constraint_loss,
  'normalize_constraint': FLAGS.normalize_constraint,
  'reduce_at_output': FLAGS.reduce_at_output,
  'similarity_reduce': FLAGS.similarity_reduce,
  'similarity_threshold': FLAGS.similarity_threshold,
  'similarity_constraint': FLAGS.similarity_constraint,

  'target_type': FLAGS.target_type,
  'target_percent': FLAGS.target_percent,
  'only_train_target': FLAGS.only_train_target,
  'ere_split': FLAGS.ere_split,

  'mapping_file': FLAGS.mapping_file,
  'embedding_dist': FLAGS.embedding_dist,
  'similarity': FLAGS.similarity,
  'augment': FLAGS.augment,
  'augment_ace05': FLAGS.augment_ace05,
  'augment_ere': FLAGS.augment_ere,
  'augment_soft_label': FLAGS.augment_soft_label,
  
  'ksupport_size': FLAGS.ksupport_size,
  'kshot_combine': FLAGS.kshot_combine,
  'ace05_kshot': FLAGS.ace05_kshot,
  'ere_kshot': FLAGS.ere_kshot,

  'rank_pos_margin': FLAGS.rank_pos_margin,
  'rank_neg_margin': FLAGS.rank_neg_margin,
  'rank_scale': FLAGS.rank_scale,
}

def get_expt_file_name():
  return (FLAGS.expt + '_' + FLAGS.model + '_' 
      + str(FLAGS.percent_train) + 'train')

def get_model_dir(expt_file_name):
  model_dir = FLAGS.model_dir +'/' + expt_file_name
  if not os.path.exists(model_dir):
    os.mkdir(model_dir)
  model_dir =  tempfile.mkdtemp(dir=model_dir)
  return model_dir

def main(argv=None):
  params = {
    'ace05': params_ace05,
    'ere': params_ere,
    'ace05_ere': params_ace05_ere,
  }
  expt = {
    'ace05': experiment_ace05,
    'ere': experiment_ere,
    'ace05_ere': experiment_ace05_ere,
  }
  expt_fn = expt[FLAGS.expt]
  expt_params = params[FLAGS.expt]
  expt_filename = get_expt_file_name()
  expt_params['model_dir'] = get_model_dir(expt_filename)
  output_filename = (FLAGS.expt + '/score' + expt_filename 
      + '_' + FLAGS.train_mode + '_' + FLAGS.joint + '_'+ FLAGS.embedding_dist
      + '_' + FLAGS.relation_type_reader + str(FLAGS.relation_embed_size))
  if FLAGS.expt.startswith('nyt10'):
    output_filename += ('_' + FLAGS.encoder + '_' + FLAGS.decoder + '_' 
        + FLAGS.mil_reduce + '_' + str(FLAGS.reduce_at_output))
  if 'target_type' in expt_params and expt_params['target_type'] != '':
    output_filename += '_' + expt_params['target_type'].replace('/','_') + str(expt_params['target_percent']) + '_'
  if FLAGS.joint.startswith('joint'):
    if FLAGS.dynamic_lamda:
      output_filename += 'd'
    else:
      output_filename += 'l'
    output_filename += str(FLAGS.lamda)
    
  if FLAGS.beta != 0.0 or FLAGS.augment != 0.0:
#    output_filename += '_beta' + str(FLAGS.beta) + '_sim' + str(FLAGS.similarity)
    output_filename += ('_beta' + str(FLAGS.beta) + '_pos' + str(FLAGS.rank_pos_margin)
                        + '_neg' + str(FLAGS.rank_neg_margin) + '_rho' + str(FLAGS.rank_scale))
  if FLAGS.augment != 0.0 or FLAGS.augment_ace05 != 0.0 or FLAGS.augment_ere != 0.0 or FLAGS.augment_nyt10 != 0.0:
    output_filename += '_aug_' + FLAGS.augment_soft_label
    if  FLAGS.augment != 0.0:
      output_filename += str(FLAGS.augment)
    if FLAGS.augment_ace05 != 0.0:
      output_filename += 'ace05' + str(FLAGS.augment_ace05)
    if FLAGS.augment_nyt10 != 0.0:
      output_filename += 'nyt10' + str(FLAGS.augment_nyt10)
    if FLAGS.augment_ere != 0.0:
      output_filename  += 'ere' + str(FLAGS.augment_ere)
  if FLAGS.constraint_loss != 0.0:
    output_filename += '_c' + str(FLAGS.constraint_loss)
  if FLAGS.da:
    output_filename += '_da' + str(FLAGS.alpha3) +FLAGS.da_input
  if FLAGS.l2:
    output_filename += '_l2' + str(FLAGS.alpha3)
  if FLAGS.meta_classifier:
    output_filename += '_meta' + str(FLAGS.alpha_meta) + FLAGS.meta_input
  if FLAGS.lr_decay:
    output_filename += '_lr_decay' + str(FLAGS.lr_decay_epoch)
  if not FLAGS.directional:
    output_filename += '_undirect'
  if FLAGS.joint == 'joint_kshot' or FLAGS.joint == 'joint_ksupport':
    output_filename += '_ksupport'+ str(FLAGS.ksupport_size)
  if FLAGS.ace05_kshot != 0:
    output_filename += '_ace05_kshot'+ str(FLAGS.ace05_kshot)
  if FLAGS.ere_kshot != 0:
    output_filename += '_ere_kshot'+ str(FLAGS.ere_kshot)
  if FLAGS.tag != '':
    output_filename += '_' + FLAGS.tag
  expt_params['score_file'] = FLAGS.score_path + '/' + output_filename
  expt_params['output_file'] = FLAGS.pred_path + '/' + output_filename
  expt_fn(expt_params)

if __name__ == '__main__':
  tf.app.run()
