
import tensorflow as tf
import numpy as np
import math

import utils
import ace05

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib import layers
from functools import partial


def embedding(incoming, input_dim, output_dim, validate_indices=False,
              weights_init=None, trainable=True, name="W"):
  # with tf.device('/cpu:0'):
  if isinstance(weights_init, tf.Tensor):
    embed = tf.get_variable(name, initializer=weights_init, trainable=trainable)
  else:
    embed = tf.get_variable(name, shape=[input_dim, output_dim],
                              initializer=weights_init, trainable=trainable)
  output = tf.nn.embedding_lookup(embed, incoming)
  return output


def embed_layer(inputs, vocab_size, max_length, entity_type_size, word_embed_dim, 
                pos_embed_dim, pretrained_embed, tune_word_embed):
  word, entity_type, pos1, pos2 = inputs
  init = tf.random_uniform_initializer(minval=-0.25, maxval=0.25)
  if pretrained_embed is None:
      word_embed = embedding(word, input_dim=vocab_size,
                                   output_dim=word_embed_dim,
                                   weights_init=init, trainable=tune_word_embed,
                                   name="word_embed")
  else:
    word_embed = embedding(word, input_dim=vocab_size,
                                 output_dim=word_embed_dim,
                                 weights_init=pretrained_embed, trainable=tune_word_embed,
                                 name="word_embed")
  if entity_type is not None:
    type_embed = embedding(entity_type, input_dim=entity_type_size,
                                   output_dim=50,
                                   weights_init=init,
                                   name="type_embed")
  pos1_embed = embedding(pos1, input_dim=max_length*2,
                                 output_dim=pos_embed_dim,
                                 weights_init=init,
                                 name="pos1_embed")
  pos2_embed = embedding(pos2, input_dim=max_length*2,
                                 output_dim=pos_embed_dim,
                                 weights_init=init,
                                 name="pos2_embed")
  if entity_type is None:
    embed = tf.concat([word_embed,pos1_embed, pos2_embed], axis=-1) #TODO
  else:
    embed = tf.concat([word_embed, type_embed,
               pos1_embed, pos2_embed
               ], axis=-1)
  return embed


def embed_layer_w2v_dropout(inputs, vocab_size, max_length, entity_type_size, word_embed_dim, 
                pos_embed_dim, pretrained_embed, tune_word_embed, mode, dropout_rate):
  word, entity_type, pos1, pos2 = inputs
  init = tf.random_uniform_initializer(minval=-0.25, maxval=0.25)
  if pretrained_embed is None:
      word_embed = embedding(word, input_dim=vocab_size,
                                   output_dim=word_embed_dim,
                                   weights_init=init, trainable=tune_word_embed,
                                   name="word_embed")
  else:
    word_embed = embedding(word, input_dim=vocab_size,
                                 output_dim=word_embed_dim,
                                 weights_init=pretrained_embed, trainable=tune_word_embed,
                                 name="word_embed")
  if mode == tf.estimator.ModeKeys.TRAIN:
      word_embed = tf.nn.dropout(word_embed, dropout_rate)
  if entity_type is not None:
    type_embed = embedding(entity_type, input_dim=entity_type_size,
                                   output_dim=50,
                                   weights_init=init,
                                   name="type_embed")
  pos1_embed = embedding(pos1, input_dim=max_length*2,
                                 output_dim=pos_embed_dim,
                                 weights_init=init,
                                 name="pos1_embed")
  pos2_embed = embedding(pos2, input_dim=max_length*2,
                                 output_dim=pos_embed_dim,
                                 weights_init=init,
                                 name="pos2_embed")
  if entity_type is None:
    embed = tf.concat([word_embed,pos1_embed, pos2_embed], axis=-1) #TODO
  else:
    embed = tf.concat([word_embed, type_embed,
               pos1_embed, pos2_embed
               ], axis=-1)
  return embed


def entity_embedding(entity_type, entity_type_size):
  init = tf.random_uniform_initializer(minval=-0.25, maxval=0.25)
  return embedding(entity_type, input_dim=entity_type_size,
                   output_dim=50, weights_init=init, name="type_embed")

def on_dep_path_embedding(on_dep_path):
  init = tf.random_uniform_initializer(minval=-0.25, maxval=0.25)
  return embedding(on_dep_path, input_dim=2, output_dim=50, 
                   weights_init=init, name='on_dep_path_embed')

def chunk_embedding(chunks, chunk_type_size):
  init = tf.random_uniform_initializer(minval=-0.25, maxval=0.25)
  return embedding(chunks, input_dim=chunk_type_size, output_dim=50,
                   weights_init=init, name="chunk_embed")
def cnn(inputs, n_filter, filter_lengths=[2,3,4,5], max_pool=True, activation=tf.nn.relu):
  with tf.variable_scope('cnn'):
    convs = []
    for filter_length in filter_lengths:
      conv = tf.layers.conv1d(inputs, n_filter, filter_length, activation=activation)
      if max_pool:
        conv = tf.layers.max_pooling1d(conv,int(inputs.get_shape()[1]-filter_length+1) , 1) 
        conv = tf.squeeze(conv, axis=1)
      convs += [conv]
    return tf.concat(convs, axis=1)

def cnn_output(inputs, n_filter, filter_lengths=[2,3,4,5], activation=tf.nn.relu):
  with tf.variable_scope('cnn_same'):
    convs = []
    for filter_length in filter_lengths:
      conv = tf.layers.conv1d(inputs, n_filter, filter_length, activation=activation, padding='same')
      print(conv)
      convs += [conv]
    return tf.concat(convs, axis=2)

def rnn(inputs, rnn_state_size):
  with tf.variable_scope('rnn'):
    input_sequence = tf.unstack(inputs, axis=1)
    fw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    outputs, states = tf.nn.static_rnn(
        fw_cell, input_sequence, dtype=tf.float32)
  
  return states

def bi_rnn_output(inputs, rnn_state_size):
  with tf.variable_scope('bi_rnn'):
    input_sequence = tf.unstack(inputs, axis=1)
    fw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    bw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(
        fw_cell, bw_cell, input_sequence, dtype=tf.float32)
    outputs = tf.stack(outputs, axis=1)
  return outputs

def bi_rnn(inputs, rnn_state_size):
  with tf.variable_scope('bi_rnn'):
    input_sequence = tf.unstack(inputs, axis=1)
    fw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    bw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(
        fw_cell, bw_cell, input_sequence, dtype=tf.float32)
  
  return tf.concat([output_state_bw, output_state_bw], axis=1)
  # # max
  #   outputs = tf.reduce_max(tf.stack(outputs, axis=2), axis=2)
  # return outputs

def word_level_product_attention(word_outputs, word_context_dim):
  """
  Args:
       word_outputs: [batch_size, rnn_output_dim] * num_word
       word_context_dim: features_dim
  Returns:
       weighted_sum: [batch_size, rnn_output_dim]
  """
  with tf.variable_scope('word_attention'):
    word_context_vector = tf.get_variable('context_vector', shape=[word_context_dim])
    word_outputs = tf.stack(word_outputs, axis=1)
  word_context_vector = tf.expand_dims(word_context_vector, axis=0)  # batch_dim
  word_context_vector = tf.expand_dims(word_context_vector, axis=1)  # num_word dim
  weight = tf.reduce_sum(tf.multiply(word_outputs, word_context_vector), axis=2) # dot product
  weight = tf.nn.softmax(weight) # normalized similarity
  weight = tf.expand_dims(weight, axis=2) # feature dim
  weighted_sum = tf.reduce_sum(tf.multiply(word_outputs, weight), axis=1) # reduce word dim
  return weighted_sum, weight

def word_level_additive_attention(word_outputs, word_context_dim):
  """
  Args:
       word_outputs: [batch_size, rnn_output_dim] * num_word
       word_context_dim: features_dim
  Returns:
       weighted_sum: [batch_size, rnn_output_dim]
  """
  with tf.variable_scope('word_attention'):
    word_context_vector = tf.get_variable('context_vector', shape=[word_context_dim])
    word_outputs = tf.stack(word_outputs, axis=1)
    projection_matrix = tf.get_variable('projection_matrix', shape=[word_outputs.shape[2], word_context_dim])
    sent_len = word_outputs.shape[1]
    projection = tf.reshape(word_outputs, [-1, word_outputs.shape[2]])
    projection = tf.matmul(projection, projection_matrix)
    projection = tf.nn.tanh(projection)
    projection = tf.reshape(projection, [-1, sent_len, projection.shape[1]])
    print(projection)
  word_context_vector = tf.expand_dims(word_context_vector, axis=0)  # batch_dim
  word_context_vector = tf.expand_dims(word_context_vector, axis=1)  # num_word dim
  weight = tf.reduce_sum(tf.multiply(projection, word_context_vector), axis=2) # dot product
  weight = tf.nn.softmax(weight) # normalized similarity
  weight = tf.expand_dims(weight, axis=2) # feature dim
  weighted_sum = tf.reduce_sum(tf.multiply(word_outputs, weight), axis=1) # reduce word dim
  return weighted_sum, weight


def rnn_att(inputs, rnn_state_size, word_context_dim):
  with tf.variable_scope('rnn_att'):
    input_sequence = tf.unstack(inputs, axis=1)
    fw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    outputs, states = tf.nn.static_rnn(
        fw_cell, input_sequence, dtype=tf.float32)
    weighted_sum, weight = word_level_additive_attention(outputs, word_context_dim)
    return weighted_sum


def bi_rnn_att(inputs, rnn_state_size, word_context_dim):
  with tf.variable_scope('bi_rnn_att'):
    input_sequence = tf.unstack(inputs, axis=1)
    fw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    bw_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_state_size)
    outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(
        fw_cell, bw_cell, input_sequence, dtype=tf.float32)
    weighted_sum, weight = word_level_additive_attention(outputs, word_context_dim)
    #global_step = tf.train.get_global_step()
    #weighted_sum = tf.cond(global_step > 1000, lambda: tf.Print(weighted_sum, [weight], summarize=100), lambda: weighted_sum)

    return weighted_sum

def encoder(inputs, mode, params):
  net = None
  if params['model'] == 'cnn':
    if 'kernel_size' in params:
      net = cnn(inputs, params['num_filters'], params['kernel_size'].split())
    else:
      net = cnn(inputs, params['num_filters'])
    if mode == tf.estimator.ModeKeys.TRAIN:
      net = tf.nn.dropout(net, 0.5)
  if params['model'] == 'bi_rnn':
    if mode == tf.estimator.ModeKeys.TRAIN:
      inputs = tf.nn.dropout(inputs, 0.5)
    net = bi_rnn(inputs, params['rnn_state_size'])

  if params['model'] == 'bi_rnn_att':
    net = bi_rnn_att(inputs, params['rnn_state_size'], params['word_context_dim'])

  if params['model'] == 'rnn':
    net = rnn(inputs, params['rnn_state_size'])

  if params['model'] == 'rnn_att':
    net =rnn_att(inputs, params['rnn_state_size'], params['word_context_dim'])

  if params['model'] == 'att':
    net = tf.unstack(inputs, axis=1)
    net, weight = word_level_additive_attention(net, params['word_context_dim'])
  
  # if mode == tf.estimator.ModeKeys.TRAIN:
  #   net = tf.nn.dropout(net, 0.5)
  return net


def decoder(net, labels, mode, params):
  with tf.variable_scope("decoder"):
    net = fully_connected(inputs=net,
                               num_outputs=params['hidden_layer_dim'],
                               activation_fn=tf.nn.relu)
    if mode == tf.estimator.ModeKeys.TRAIN:
      net = tf.nn.dropout(net, 0.5)
      
    relation = fully_connected(inputs=net,
                               num_outputs=params['num_classes'],
                               activation_fn=None)
  relation_prob = tf.nn.softmax(relation)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"relation": relation_prob})
    
  # train and eval
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
  loss = tf.reduce_mean(loss, 0)
  
  if params['lr_decay']:
    learning_rate_decay_fn = partial(tf.train.exponential_decay, 
                                   decay_steps=params['learning_rate_decay_step'],
                                   decay_rate=0.5, staircase=True)
  else:
    learning_rate_decay_fn = None
  
  train_op = layers.optimize_loss(
      loss, tf.train.get_global_step(),
      learning_rate_decay_fn=learning_rate_decay_fn,
      learning_rate=params['learning_rate'],
      optimizer='Adam')

  if mode == 'train':
    print(utils.variable_shapes())

  
  return tf.estimator.EstimatorSpec(mode, {"relation": relation_prob}, loss, train_op)


def relation_stack_model(features, labels, mode, params):
  inputs = features['word'], None, features['pos1'], features['pos2']
  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder"):
    embed = embed_layer(inputs, params['vocab_size'], params['max_len'], 
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    if 'type' in features:
      entity_embed = entity_embedding(features['type'], params['entity_type_size'])
      embed = tf.concat([embed, entity_embed], axis=-1)

    with tf.variable_scope('hidden'):
      hidden = bi_rnn_output(embed, params['hidden_rnn_size'])
    if params['extra_feature']:
      extra_embed = None
      if 'on_dep_path' in features:
        print('Using on_dep_path')
        extra_embed = on_dep_path_embedding(features['on_dep_path'])
      if 'chunk' in features:
        print('Using chunk')
        chunk_embed = chunk_embedding(features['chunk'], params['chunk_type_size'])
      if extra_embed is None:
        extra_embed = chunk_embed
      else:
        extra_embed = tf.concat([extra_embed, chunk_embed], axis=-1)
      hidden = tf.concat([hidden, extra_embed], axis=-1)
    net = encoder(hidden, mode, params)
  return decoder(net, labels, mode, params)


def relation_model(features, labels, mode, params):
  if 'type' in features:
    inputs = features['word'], features['type'], features['pos1'], features['pos2']
  else:
    inputs = features['word'], None, features['pos1'], features['pos2']
  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder"):
    if params['input_dropout'] > 0:
      net = embed_layer(inputs, params['vocab_size'], params['max_len'], 
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'], mode, params['input_dropout'])
    else:
      net = embed_layer(inputs, params['vocab_size'], params['max_len'], 
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    if params['extra_feature']:
      extra_embed = None
      if 'on_dep_path' in features:
        print('Using on_dep_path')
        extra_embed = on_dep_path_embedding(features['on_dep_path'])
      if 'chunk' in features:
        print('Using chunk')
        chunk_embed = chunk_embedding(features['chunk'], params['chunk_type_size'])
      if extra_embed is None:
        extra_embed = chunk_embed
      else:
        extra_embed = tf.concat([extra_embed, chunk_embed], axis=-1)
      net = tf.concat([net, extra_embed], axis=-1)
    net = encoder(net, mode, params)    
  return decoder(net, labels, mode, params)



def relation_type_encoder(relation_name, word_embed, num_classes, params, scope=''):
  r_name_embed = tf.nn.embedding_lookup(word_embed, relation_name)
  if params['relation_type_reader'] == 'mean':
    r_repr = tf.reduce_mean(r_name_embed, axis=1)
    r_repr = fully_connected(inputs=r_repr,
                             num_outputs=params['relation_embed_size'],
                             activation_fn=tf.nn.tanh, scope='relation_type_reader')
  elif params['relation_type_reader'] == 'max':
    r_repr = tf.reduce_max(r_name_embed, axis=1)
    r_repr = fully_connected(inputs=r_repr,
                             num_outputs=params['relation_embed_size'],
                             activation_fn=tf.nn.tanh, scope='relation_type_reader')
  elif params['relation_type_reader'] == 'sum':
    r_repr = tf.reduce_sum(r_name_embed, axis=1)
    r_repr = fully_connected(inputs=r_repr,
                             num_outputs=params['relation_embed_size'],
                             activation_fn=tf.nn.tanh, scope='relation_type_reader')
  elif params['relation_type_reader'] == 'shared':
    with tf.variable_scope('encoder/hidden', reuse=True):
      r_repr =  bi_rnn(r_name_embed, params['hidden_rnn_size'])
  elif params['relation_type_reader'] == 'bi_rnn':
    with tf.variable_scope('encoder/relation_type'):
      r_repr = bi_rnn(r_name_embed, params['relation_embed_size'])
  elif params['relation_type_reader'] == 'rnn':
    with tf.variable_scope('encoder/relation_type'):
      r_repr = rnn(r_name_embed, params['relation_embed_size'])
  elif params['relation_type_reader'] == 'cnn':
    with tf.variable_scope('encoder/relation_type'):
      r_repr = cnn(r_name_embed, params['relation_embed_size'], filter_lengths=[3], activation=tf.nn.tanh)
  elif params['relation_type_reader'] == 'random':
    r = math.sqrt(6/(num_classes+params['relation_embed_size']))
#    init = tf.random_uniform_initializer(minval=-r, maxval=r)
    with tf.variable_scope('encoder/relation_type/'+scope):
      r_repr = tf.get_variable('r_repr', shape=[num_classes, params['relation_embed_size']], 
                               trainable=True)
  if params['r_repr'] == 'extra':
    reprs_combined = []
    for i in xrange(r_repr.shape[0]):
      other_repr = tf.reduce_mean(tf.concat([r_repr[:i], r_repr[i+1:]], axis=0), axis=0)
      reprs_combined += [tf.concat([r_repr[i,:], other_repr], axis=0)]
    r_repr = tf.stack(reprs_combined, axis=0)
  return r_repr