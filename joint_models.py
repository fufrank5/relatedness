import tensorflow as tf
import numpy as np
from models import (embed_layer, encoder, entity_embedding, on_dep_path_embedding, chunk_embedding, 
  word_level_additive_attention, bi_rnn_output, relation_type_encoder)
import utils
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib import layers
from functools import partial

def pretrain_relation_model(features, labels, mode, params):
  if 'type' in features:
    inputs = features['word'], features['type'], features['pos1'], features['pos2']
  else:
    inputs = features['word'], None, features['pos1'], features['pos2']
  if params['pretrain_w2v']:
    pretrained_embed = tf.to_float(tf.constant(params['embed']))
  else:
    pretrained_embed = None

  if params['current_dataset'] == 'ace05':
    num_classes = params['ace05_num_classes']
  elif params['current_dataset'] == 'ere':
    num_classes = params['ere_num_classes']

  with tf.variable_scope("encoder"):
    net = embed_layer(inputs, params['vocab_size'], params['max_len'], 
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    
    net = encoder(net, mode, params)
  if params['pretrained_model'] is not None and params['current_epoch'] == 1 and mode == tf.estimator.ModeKeys.TRAIN:
    tf.contrib.framework.init_from_checkpoint(params['pretrained_model'], {'encoder/': 'encoder/'})
  with tf.variable_scope(params['current_dataset']):
    relation_prob, loss = decoder(net, labels, num_classes,params['num_hidden_layer'], params['hidden_layer_dim'], mode)
  if params['train_mode'] == 'decoder':
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=params['current_dataset'])
  else:
    variables = None  # All trainable variables
  return make_estimator_spec(relation_prob, loss, mode, params, variables)

def joint_bi_rnn_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None

  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as encoder_scope:
    with tf.variable_scope('hidden') as hidden_scope:
      ace05_embed = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                                params['entity_type_size'], params['word_embed_dim'],
                                params['pos_embed_dim'],
                                pretrained_embed, params['tune_word_embed'])
      ace05_hidden = bi_rnn_output(ace05_embed, params['hidden_rnn_size'])
      hidden_scope.reuse_variables()
      ere_embed = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ere_hidden = bi_rnn_output(ere_embed, params['hidden_rnn_size'])

    if params['extra_feature']:
      extra_embed = None
      if 'ace05_on_dep_path' in features:
        print('Using on_dep_path')
        extra_embed = on_dep_path_embedding(features['ace05_on_dep_path'])
      if 'ace05_chunk' in features:
        print('Using chunk')
        chunk_embed = chunk_embedding(features['ace05_chunk'], params['chunk_type_size'])
        if extra_embed is not None:
          extra_embed = tf.concat([extra_embed, chunk_embed], axis=-1)
        else:
          extra_embed = chunk_embed
      ace05_hidden = tf.concat([ace05_hidden, extra_embed], axis=-1)

  with tf.variable_scope('ace05'):
      #ace05_net = encoder(ace05_hidden, mode, params)
      ace05_hidden = tf.unstack(ace05_hidden, axis=1)
      ace05_net, ace05_weight = word_level_additive_attention(ace05_hidden, params['word_context_dim'])
      ace05_prob, ace05_loss = decoder(ace05_net, ace05_labels, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)

  with tf.variable_scope('ere'):
#      ere_net = encoder(ere_hidden, mode, params)
      ere_hidden = tf.unstack(ere_hidden, axis=1)
      ere_net, ere_weight = word_level_additive_attention(ere_hidden, params['word_context_dim'])
      ere_prob, ere_loss = decoder(ere_net, ere_labels, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)

  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, None, mode, params)


def joint_stack_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None

  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as encoder_scope:
    with tf.variable_scope('hidden') as hidden_scope:
      ace05_embed = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                                params['entity_type_size'], params['word_embed_dim'],
                                params['pos_embed_dim'],
                                pretrained_embed, params['tune_word_embed'])
      ace05_hidden = bi_rnn_output(ace05_embed, params['hidden_rnn_size'])
      hidden_scope.reuse_variables()
      ere_embed = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ere_hidden = bi_rnn_output(ere_embed, params['hidden_rnn_size'])
    if params['da']:
      ere_da_input = tf.reduce_max(ere_hidden, axis=1)
      ace05_da_input = tf.reduce_max(ace05_hidden, axis=1)
      adv_loss =  add_adversarial_loss(ace05_da_input, ere_da_input)
    else:
      adv_loss = None
    if params['extra_feature']:
      extra_embed = None
      if 'ace05_on_dep_path' in features:
        print('Using on_dep_path')
        extra_embed = on_dep_path_embedding(features['ace05_on_dep_path'])
      if 'ace05_chunk' in features:
        print('Using chunk')
        chunk_embed = chunk_embedding(features['ace05_chunk'], params['chunk_type_size'])
        if extra_embed is not None:
          extra_embed = tf.concat([extra_embed, chunk_embed], axis=-1)
        else:
          extra_embed = chunk_embed      
      ace05_hidden = tf.concat([ace05_hidden, extra_embed], axis=-1)
    
  with tf.variable_scope('ace05'):
      ace05_net = encoder(ace05_hidden, mode, params)
      ace05_prob, ace05_loss = decoder(ace05_net, ace05_labels, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)

  with tf.variable_scope('ere'):
      ere_net = encoder(ere_hidden, mode, params)
      ere_prob, ere_loss = decoder(ere_net, ere_labels, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)

  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, adv_loss, mode, params)


def joint_repr_model(features, labels, mode, params):
  print(features)
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None

  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as encoder_scope:
    with tf.variable_scope('hidden') as hidden_scope:
      ace05_embed = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                                params['entity_type_size'], params['word_embed_dim'],
                                params['pos_embed_dim'],
                                pretrained_embed, params['tune_word_embed'])
      ace05_hidden = bi_rnn_output(ace05_embed, params['hidden_rnn_size'])
      ace05_net = encoder(ace05_hidden, mode, params)
      hidden_scope.reuse_variables()
      ere_embed = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ere_hidden = bi_rnn_output(ere_embed, params['hidden_rnn_size'])
      ere_net = encoder(ere_hidden, mode, params)
#    if params['extra_feature']:
#      extra_embed = None
#      if 'ace05_on_dep_path' in features:
#        print('Using on_dep_path')
#        extra_embed = on_dep_path_embedding(features['ace05_on_dep_path'])
#      if 'ace05_chunk' in features:
#        print('Using chunk')
#        chunk_embed = chunk_embedding(features['ace05_chunk'], params['chunk_type_size'])
#        if extra_embed is not None:
#          extra_embed = tf.concat([extra_embed, chunk_embed], axis=-1)
#        else:
#          extra_embed = chunk_embed      
#      ace05_hidden = tf.concat([ace05_hidden, extra_embed], axis=-1)

  with tf.variable_scope('relation_type') as type_scope:
    names_embed = tf.get_variable('label_name_embed', dtype=tf.float32,
                                  initializer=params['relation_names_embed'], trainable=True)
    ace05_r_repr = relation_type_encoder(params['ace05_relation_names'], names_embed, params['ace05_num_classes'], params, scope='ace05')
#    if params['constraint_loss'] != 0.0:
#      relation_embedding_constraint(ace05_r_repr, scope='ace05')
    if params['relation_type_reader'] != 'random':
      type_scope.reuse_variables()
    ere_r_repr = relation_type_encoder(params['ere_relation_names'], names_embed, params['ere_num_classes'], params, scope='ere')
#    if params['relation_type_reader'] == 'random':
#      ace05_r_repr = tf.concat([ere_r_repr[0:1,:], ace05_r_repr[1:,:]], axis=0) # same other as ere
   # if params['constraint_loss'] != 0.0:
  with tf.variable_scope('relation_type/meta'):  
    # relation_embedding_constraint(ace05_r_repr, scope='ace05')
    # relation_embedding_constraint(ere_r_repr, scope='ere')
    
#    if params['embedding_dist'] == 'cosine':
    r = get_type_correlation(ere_r_repr, ace05_r_repr)
#    elif params['embedding_dist'] == 'l1':
#    r = get_l1_distance(ere_r_repr, ace05_r_repr)
    # r = get_l2_distance(ere_r_repr, ace05_r_repr)
    r_output = tf.get_variable('similarity', initializer=r, trainable=False)
    output_assign = tf.assign(r_output, r)
    with tf.control_dependencies([output_assign]):
      ace05_net = tf.identity(ace05_net)
#      ere_r_repr = tf.Print(ere_r_repr, [r], message='correlation', summarize=ace05_r_repr.shape[0]*ere_r_repr.shape[0])
    # other do not share similarity

   # positive_indices = get_ones_indices(params['ere2ace05_index'])
#    mapping_indices = [[0,0]]
#    if params['embedding_dist'] == 'cosine':
#    correlation_loss = correlation_constraint_loss(tf.gather_nd(r[1:, 1:], positive_indices), tf.gather_nd(params['ere2ace05_index'][1:,1:], positive_indices))
#    correlation_loss = correlation_constraint_loss(r[1:, 1:], params['ere2ace05_index'][1:, 1:])
#    correlation_loss = r[0, 0]
#    elif params['embedding_dist'] == 'l1':
#    correlation_loss = get_l1_distance_loss(tf.gather_nd(r[1:, 1:], positive_indices))
#    correlation_loss = tf.reduce_mean(tf.gather_nd(r, mapping_indices))
  DECODER_FN = {
    'joint_repr': repr_decoder,
    'joint_repr_NONE': repr_decoder,
    'joint_embedding': label_embedding_decoder,
    'joint_reader': reader_decoder,
  }
  decoder = DECODER_FN[params['joint']]
  if params['joint'] == 'joint_repr_NONE':
    ace05_r_repr, ere_r_repr = None, None
  adv_loss, aug_loss = None, None
  if params['joint'] == 'joint_repr' or params['joint'] == 'joint_repr_NONE':
    with tf.variable_scope('ace05'):
      ace05_loss, ace05_prob = decoder(ace05_net, ace05_labels, mode,
          params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax')
    with tf.variable_scope('ere'):  
      ere_loss, ere_prob = decoder(ere_net, ere_labels, mode,
          params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax')
    correlation_loss = None
  else:
    with tf.variable_scope('ace05_ere') as scope:
      ace05_net = fully_connected(inputs=ace05_net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
      scope.reuse_variables()
      ere_net = fully_connected(inputs=ere_net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
    with tf.variable_scope('ace05_ere') as scope:
      _, ere2ace05_prob = decoder(ere_r_repr, None, None,  params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax', weights=None)
      scope.reuse_variables()
      _, ace05_self_prob = decoder(ace05_r_repr, None, None,  params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax', weights=None)
      _, ere_self_prob = decoder(ere_r_repr, None, None,  params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax', weights=None)
    with tf.variable_scope('relation_type/meta'):
      relation_embedding_constraint(ace05_self_prob, scope='ace05')
      relation_embedding_constraint(ere_self_prob, scope='ere')
    with tf.variable_scope('relation_type/meta'):
      p_output = tf.get_variable('ere2ace05_prob', initializer=ere2ace05_prob, trainable=False)
#      positive_indices = get_ones_indices(params['ere2ace05_index'])
#      correlation_loss = tf.reduce_mean(tf.gather_nd(r, positive_indices))
      # correlation_loss = correlation_constraint_loss(tf.gather_nd(ere2ace05_prob[1:,1:], positive_indices),
                                                   # tf.gather_nd(params['ere2ace05_index'][1:,1:], positive_indices))
      # correlation_loss = correlation_constraint_loss(r[1:,1:], params['ere2ace05_index'][1:,1:])
      output_assign = tf.assign(p_output, ere2ace05_prob)
      with tf.control_dependencies([output_assign]):
       ace05_net = tf.identity(ace05_net)
      
    with tf.variable_scope('ace05_ere', reuse=True) as scope:
      if mode == tf.estimator.ModeKeys.TRAIN and params['beta'] > 0.0:
        p_ace05_net, p_ace05_labels, p_ere_net, p_ere_labels = add_pseudo_labels(
            ace05_net, ace05_labels, params['augment_ace05'], ere_net, ere_labels, params['augment_ere'], params['ere2ace05_index'], params['ere2ace05_index'], params)         
        # ace05_net, ace05_labels, ace05_weights = concat_pseudo_labels(ace05_net, ace05_labels, p_ace05_net, p_ace05_labels, params['augment_ace05'] if params['augment_ace05'] > 0.0 else params['augment'])
        # ere_net, ere_labels, ere_weights = concat_pseudo_labels(ere_net, ere_labels, p_ere_net, p_ere_labels, params['augment_ere'] if params['augment_ere'] > 0.0 else params['augment'])
        p_ace05_loss, p_ace05_prob = decoder(p_ace05_net, p_ace05_labels, mode, 
          params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='ranking')
        p_ere_loss, p_ere_prob = decoder(p_ere_net, p_ere_labels, mode, 
          params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='ranking')
        # correlation_loss = p_ace05_loss * params['augment_ace05'] + p_ere_loss * params['augment_ere']
        correlation_loss = p_ace05_loss

      ace05_loss, ace05_prob = decoder(ace05_net, ace05_labels, mode, 
          params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax')
      ere_loss, ere_prob = decoder(ere_net, ere_labels, mode, 
          params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax')
      

  if params['da']:
    adv_loss =  add_adversarial_loss(ace05_net, ere_net)
  if params['l2']: #cannot be used together with da
    adv_loss =  add_l2_loss(ace05_net, ere_net)
  if params['pretrained_model'] != "" and params['current_epoch'] == 1 and mode == tf.estimator.ModeKeys.TRAIN:
    tf.contrib.framework.init_from_checkpoint(params['pretrained_model'], {'encoder/': 'encoder/', 'ace05_ere/': 'ace05_ere/'})
  if params['pretrained_model'] != "":
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='relation_type/encoder/relation_type/')
  else:
    variables = None
  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, adv_loss, mode, params, variables=variables,
                             similarity_loss=correlation_loss)


def relation_embedding_constraint(self_prob, scope, ref=None):
  """take the diagnal"""
  with tf.variable_scope(scope):
    c = tf.get_variable('self_constraint', shape=self_prob.shape, trainable=False)
    assign_c = tf.assign(c, self_prob)
  with tf.control_dependencies([assign_c]):
    self_prob = tf.identity(self_prob)
  loss = 0.5*tf.reduce_mean(tf.square(self_prob - tf.eye(num_rows=int(self_prob.shape[0])) if ref is None else ref))
  tf.summary.scalar('r_repr_constraint/'+ scope, loss)
  tf.add_to_collection('r_repr_constraint', loss)
  return loss


def add_pseudo_labels(ace05_net, ace05_labels, augment_ace05, ere_net, ere_labels, augment_ere, label_mapping, r, params):
	convert_m = tf.cast(label_mapping, tf.float32)
        #pseudo ere
        pseudo_ere_labels = tf.matmul(a=ace05_labels, b=convert_m, transpose_b=True)
        pos_indices = tf.squeeze(tf.where(tf.not_equal(tf.argmax(pseudo_ere_labels, axis=1),0)), axis=1)
        pseudo_ere_net = tf.gather(ace05_net, pos_indices, axis=0)
        pseudo_ere_labels = tf.gather(ace05_labels, pos_indices, axis=0)

        print('pseudo ere pos', pos_indices)
        if params['augment_soft_label'] == 'positive':
          # use similarity as prob, 1-prob for other
          pseudo_ere_labels = tf.matmul(a=pseudo_ere_labels, b=tf.multiply(convert_m, r), transpose_b=True)
          other_prob = tf.expand_dims(1.0 - tf.reduce_sum(pseudo_ere_labels, axis=1), axis=1)
          pseudo_ere_labels = tf.concat([other_prob, pseudo_ere_labels[:, 1:]], axis=1)
        elif params['augment_soft_label'] == 'norm':
          pseudo_ere_labels = tf.matmul(pseudo_ere_labels, r, transpose_b=True)
          indices = tf.squeeze(tf.where(tf.greater(tf.reduce_sum(tf.nn.relu(pseudo_ere_labels), axis=1), 0.0)), axis=1) # avoid nan loss when all negative
          pseudo_ere_labels = tf.gather(pseudo_ere_labels, indices, axis=0)
          # normalize prob
          pseudo_ere_labels = tf.nn.relu(pseudo_ere_labels)/tf.expand_dims(tf.reduce_sum(tf.nn.relu(pseudo_ere_labels), axis=1), axis=1)

        elif params['augment_soft_label'] == 'softmax':
          pseudo_ere_labels = tf.matmul(pseudo_ere_labels, r, transpose_b=True)
          pseudo_ere_labels = tf.nn.softmax(pseudo_ere_labels)
        elif params['augment_soft_label'] == 'direct':
          pseudo_ere_labels = tf.matmul(pseudo_ere_labels, r, transpose_b=True)
        pseudo_ere_labels = tf.stop_gradient(pseudo_ere_labels) #?
        
        #pseudo ace05
        pseudo_ace05_labels = tf.matmul(a=ere_labels, b=convert_m)
        pos_indices = tf.squeeze(tf.where(tf.not_equal(tf.argmax(pseudo_ace05_labels, axis=1),0)), axis=1)
        print('pseudo ace05 pos', pos_indices)
        pseudo_ace05_net = tf.gather(ere_net, pos_indices, axis=0)
        pseudo_ace05_labels = tf.gather(ere_labels, pos_indices, axis=0)
        if params['augment_soft_label'] == 'positive':
          # use similarity as prob, 1-prob for other
          pseudo_ace05_labels = tf.matmul(a=pseudo_ace05_labels, b=tf.multiply(convert_m, r))
          other_prob = tf.expand_dims(1.0 - tf.reduce_sum(pseudo_ace05_labels, axis=1), axis=1)
          pseudo_ace05_labels = tf.concat([other_prob, pseudo_ace05_labels[:, 1:]], axis=1)
        elif params['augment_soft_label'] == 'norm':
          print(pseudo_ace05_labels)
          pseudo_ace05_labels = tf.matmul(pseudo_ace05_labels, r)
          print(pseudo_ace05_labels)
          pseudo_ace05_labels = tf.nn.relu(pseudo_ace05_labels)/tf.expand_dims(tf.reduce_sum(tf.nn.relu(pseudo_ace05_labels), axis=1), axis=1)
        elif params['augment_soft_label'] == 'softmax':
          pseudo_ace05_labels = tf.matmul(pseudo_ace05_labels, r)
          pseudo_ace05_labels = tf.nn.softmax(pseudo_ace05_labels)
        elif params['augment_soft_label'] == 'direct':
          pseudo_ace05_labels = tf.matmul(pseudo_ace05_labels, r)
        pseudo_ace05_labels = tf.stop_gradient(pseudo_ace05_labels)
        # pseudo_ace05_net = tf.stop_gradient(pseudo_ace05_net)
        return pseudo_ace05_net, pseudo_ace05_labels, pseudo_ere_net, pseudo_ere_labels
 
def concat_pseudo_labels(ace05_net, ace05_labels, pseudo_ace05_net, pseudo_ace05_labels, augment):
        # add pseudo ace05
        ace05_weights = tf.concat([tf.fill([tf.shape(ace05_net)[0]],1.0),
                                   tf.fill([tf.shape(pseudo_ace05_net)[0]],
                                           augment)],
                                  axis=0)
        ace05_net = tf.concat([ace05_net, pseudo_ace05_net], axis=0)
        ace05_labels = tf.concat([ace05_labels, pseudo_ace05_labels], axis=0)

        return ace05_net, ace05_labels, ace05_weights

def get_type_correlation(r_repr1, r_repr2):
  """normalized correlation based on cosine"""
#  norm1 = tf.norm(r_repr1, axis=1, keep_dims=True)
#  norm2 = tf.norm(r_repr2, axis=1, keep_dims=True)
#  norm = tf.matmul(norm1, tf.transpose(norm2, perm=[1, 0]))

  dot_product = tf.matmul(r_repr1, tf.transpose(r_repr2, perm=[1, 0]))
#  r = tf.div(dot_product, norm)
#  r = tf.nn.sigmoid(dot_product)
  return dot_product

def correlation_constraint_loss(r, mapping_matrix):
#  loss = tf.losses.sigmoid_cross_entropy(logits=r, multi_class_labels=mapping_matrix, reduction=tf.losses.Reduction.NONE)
#  return tf.reduce_mean(loss) #cross entropy loss
 return 0.5*tf.reduce_mean(tf.square(mapping_matrix-r)) # mean square error

def correlation_cross_entropy_loss(r, mapping_matrix, loss_fn='sigmoid'):
  if loss_fn == 'sigmoid':
    loss = tf.losses.sigmoid_cross_entropy(logits=r, multi_class_labels=mapping_matrix, reduction=tf.losses.Reduction.MEAN)
  else:
    loss = tf.losses.softmax_cross_entropy(logits=r, multi_class_labels=mapping_matrix, reduction=tf.losses.Reduction.MEAN)
  return loss


def correlation_ranking_loss(r, mapping_matrix, pos_margin=2.5, neg_margin=0.5, scale=2):
  zero = tf.constant(0, dtype=tf.float32)
  neg_values = tf.where(tf.equal(mapping_matrix, zero), r, tf.fill(r.shape, neg_margin-10))
  pos_values = tf.where(tf.not_equal(mapping_matrix, zero), r, tf.fill(r.shape, pos_margin+10))
  neg_values = tf.reduce_max(neg_values, axis=1)
  pos_values = tf.reduce_min(pos_values, axis=1)
#  loss = (tf.nn.relu(tf.multiply(tf.subtract(pos_margin, pos_values), scale)) 
#          + tf.nn.relu(tf.multiply(tf.add(neg_margin, neg_values), scale)))
  loss = (tf.log(1+tf.exp(tf.multiply(tf.subtract(pos_margin, pos_values), scale))) 
          + tf.log(1+ tf.exp(tf.multiply(tf.add(neg_margin, neg_values), scale))))
  return tf.reduce_mean(loss, axis=0)


def ranking_loss(net, labels, pos_margin=2.5, neg_margin=0.5, scale=2):
  "same as correlation_ranking_loss, intended for instances"
  zero = tf.constant(0, dtype=tf.float32)
  neg_values = tf.where(tf.equal(labels, zero), net, tf.fill(tf.shape(net), neg_margin-10))
  pos_values = tf.where(tf.not_equal(labels, zero), net, tf.fill(tf.shape(net), pos_margin+10))
  neg_values = tf.reduce_max(neg_values, axis=1)
  pos_values = tf.reduce_min(pos_values, axis=1)
  loss = (tf.nn.relu(tf.multiply(tf.subtract(pos_margin, pos_values), scale)) 
          + tf.nn.relu(tf.multiply(tf.add(neg_margin, neg_values), scale)))
#  loss = (tf.log(1+tf.exp(tf.multiply(tf.subtract(pos_margin, pos_values), scale))) 
#          + tf.log(1+ tf.exp(tf.multiply(tf.add(neg_margin, neg_values), scale))))
  loss =  tf.reduce_mean(loss, axis=0)
  return loss

def joint_kshot_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  ace05_kshot_inputs = features['ace05_kshot_word'], features['ace05_kshot_type'], features['ace05_kshot_pos1'], features['ace05_kshot_pos2']
  ere_kshot_inputs = features['ere_kshot_word'], features['ere_kshot_type'], features['ere_kshot_pos1'], features['ere_kshot_pos2']

  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None

  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as encoder_scope:
    with tf.variable_scope('hidden') as hidden_scope:
      ace05_embed = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                                params['entity_type_size'], params['word_embed_dim'],
                                params['pos_embed_dim'],
                                pretrained_embed, params['tune_word_embed'])
      ace05_hidden = bi_rnn_output(ace05_embed, params['hidden_rnn_size'])
      ace05_net = encoder(ace05_hidden, mode, params)
      ace05_net = fully_connected(inputs=ace05_net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
      hidden_scope.reuse_variables()
      ere_embed = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ere_hidden = bi_rnn_output(ere_embed, params['hidden_rnn_size'])
      ere_net = encoder(ere_hidden, mode, params)
      ere_net = fully_connected(inputs=ere_net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
      ace05_r_repr = relation_kshot_encoder(ace05_kshot_inputs, pretrained_embed, mode, params['ace05_num_classes'], params)
      ere_r_repr = relation_kshot_encoder(ere_kshot_inputs, pretrained_embed, mode, params['ere_num_classes'], params)

  with tf.variable_scope('ace05_ere') as scope:
    ace05_loss, ace05_prob = label_embedding_decoder(ace05_net, ace05_labels, mode,
        params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax')
    scope.reuse_variables()
    ere_loss, ere_prob = label_embedding_decoder(ere_net, ere_labels, mode,
        params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax')
  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, None, mode, params)

def joint_ksupport_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  ace05_kshot_inputs = features['ace05_kshot_word'], features['ace05_kshot_type'], features['ace05_kshot_pos1'], features['ace05_kshot_pos2']
  ere_kshot_inputs = features['ere_kshot_word'], features['ere_kshot_type'], features['ere_kshot_pos1'], features['ere_kshot_pos2']

  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None

  if params['pretrain_w2v']:
    pretrained_embed = tf.constant_initializer(params['embed'], dtype=tf.float32, verify_shape=True)
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as encoder_scope:
    with tf.variable_scope('hidden') as hidden_scope:
      ace05_embed = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                                params['entity_type_size'], params['word_embed_dim'],
                                params['pos_embed_dim'],
                                pretrained_embed, params['tune_word_embed'])
      ace05_hidden = bi_rnn_output(ace05_embed, params['hidden_rnn_size'])
      ace05_net = encoder(ace05_hidden, mode, params)
      ace05_net = fully_connected(inputs=ace05_net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
      hidden_scope.reuse_variables()
      ere_embed = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ere_hidden = bi_rnn_output(ere_embed, params['hidden_rnn_size'])
      ere_net = encoder(ere_hidden, mode, params)
      ere_net = fully_connected(inputs=ere_net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
      ace05_kshot_net = relation_kshot_encoder(ace05_kshot_inputs, pretrained_embed, mode, params['ace05_num_classes'], params, reduce=False)
      ace05_kshot_repr = tf.reduce_mean(ace05_kshot_net, axis=1)
      ere_kshot_net = relation_kshot_encoder(ere_kshot_inputs, pretrained_embed, mode, params['ere_num_classes'], params, reduce=False)
      ere_kshot_repr = tf.reduce_mean(ere_kshot_net, axis=1)

  with tf.variable_scope('relation_type/meta'):  
    # relation_embedding_constraint(ace05_r_repr, scope='ace05')
    # relation_embedding_constraint(ere_r_repr, scope='ere')
    
#    if params['embedding_dist'] == 'cosine':
    r = get_type_correlation(ere_kshot_repr, ace05_kshot_repr)
#    elif params['embedding_dist'] == 'l1':
   # r = get_l1_distance(ere_kshot_repr, ace05_kshot_repr)
    # r = get_l2_distance(ere_r_repr, ace05_r_repr)
    r_output = tf.get_variable('similarity', shape=r.shape, trainable=False)
    output_assign = tf.assign(r_output, r)
    with tf.control_dependencies([output_assign]):
      ace05_net = tf.identity(ace05_net)
    # correlation_loss = correlation_cross_entropy_loss(r, params['ere2ace05_index'])
    # correlation_loss = correlation_ranking_loss(r, params['ere2ace05_index'], params['rank_pos_margin'], params['rank_neg_margin'], params['rank_scale'])

  with tf.variable_scope('relation_type') as type_scope:
    names_embed = tf.get_variable('label_name_embed', dtype=tf.float32,
                                  initializer=params['relation_names_embed'], trainable=True)
    ace05_r_repr = relation_type_encoder(params['ace05_relation_names'], names_embed, params['ace05_num_classes'], params, scope='ace05')
    if params['relation_type_reader'] != 'random':
      type_scope.reuse_variables()
    ere_r_repr = relation_type_encoder(params['ere_relation_names'], names_embed, params['ere_num_classes'], params, scope='ere')
  with tf.variable_scope('ace05_ere') as scope:
    ace05_loss, ace05_prob = label_embedding_decoder(ace05_net, ace05_labels, mode,
        params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax')
    scope.reuse_variables()
    ere_loss, ere_prob = label_embedding_decoder(ere_net, ere_labels, mode,
        params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax')
  with tf.variable_scope('ace05_ere', reuse=True) as scope:
    _, ere2ace05_net, ere2ace05_prob = label_embedding_decoder_logits(ere_kshot_repr, None, None,  params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax', weights=None)
    _, ace052ere_net, ace052ere_prob = label_embedding_decoder_logits(ace05_kshot_repr, None, None,  params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax', weights=None)

    _, ace05_self_prob = label_embedding_decoder(ace05_kshot_repr, None, None,  params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='softmax', weights=None)
    _, ere_self_prob = label_embedding_decoder(ere_kshot_repr, None, None,  params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='softmax', weights=None)
  with tf.variable_scope('relation_type/meta'):
      relation_embedding_constraint(ace05_self_prob[1:,1:], scope='ace05')
      relation_embedding_constraint(ere_self_prob[1:,1:], scope='ere')
  correlation_loss, output_ops = add_correlation_constraint(ere2ace05_prob, ace052ere_prob, params['ere2ace05_index'], ere2ace05_net, ace052ere_net, params)
  with tf.control_dependencies(output_ops):
    if ace05_loss is not None:
      ace05_loss = tf.identity(ace05_loss)   

  augment_loss = add_kshot_augment(ace05_kshot_net, ace05_r_repr, ere_kshot_net, ere_r_repr, mode, params)
#  augment_loss = add_correlation_augment(ace05_net, ace05_labels, ace05_r_repr, ere_net, ere_labels, ere_r_repr, label_embedding_decoder, mode, params)
 
  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, None, mode, params,
                             similarity_loss=correlation_loss, augment_loss=augment_loss)

def add_kshot_augment(ace05_kshot_net, ace05_r_repr, ere_kshot_net, ere_r_repr, mode, params):
  if mode != tf.estimator.ModeKeys.TRAIN or params['augment'] == 0.0:
    return None
  with tf.variable_scope('ace05_ere', reuse=True) as scope:
    ace05_net, ace05_labels = kshot_to_normal(ace05_kshot_net, params['ace05_num_classes'], params)
    ere_net, ere_labels = kshot_to_normal(ere_kshot_net, params['ere_num_classes'], params)
    convert_m = tf.cast(params['ere2ace05_index'], tf.float32)
    #pseudo labels
    p_ere_labels = tf.matmul(a=ace05_labels, b=convert_m, transpose_b=True)  
    p_ace05_labels = tf.matmul(a=ere_labels, b=convert_m)
    p_ace05_loss, p_ace05_prob = label_embedding_decoder(ere_net, p_ace05_labels, mode,
           params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='ranking')
    p_ere_loss, p_ere_prob = label_embedding_decoder(ace05_net, p_ere_labels, mode,
           params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='ranking')
    augment_loss = p_ace05_loss * params['augment_ace05'] + p_ere_loss * params['augment_ere']
    tf.summary.scalar('augment_loss', augment_loss)
    return augment_loss

def add_correlation_augment(ace05_net, ace05_labels, ace05_r_repr,
                            ere_net, ere_labels, ere_r_repr, decoder, mode, params):
  with tf.variable_scope('ace05_ere', reuse=True) as scope:
    if mode == tf.estimator.ModeKeys.TRAIN and params['augment'] > 0.0:
      p_ace05_net, p_ace05_labels, p_ere_net, p_ere_labels = add_pseudo_labels(
          ace05_net, ace05_labels, params['augment_ace05'], ere_net, ere_labels, params['augment_ere'], params['ere2ace05_index'], params['ere2ace05_index'], params)         
      if params['augment_ace05'] != 0.0:
        p_ace05_loss, p_ace05_prob = decoder(p_ace05_net, p_ace05_labels, mode, 
          params['ace05_num_classes'], params, r_repr=ace05_r_repr, loss_fn='ranking')
        p_ace05_loss = tf.cond(tf.greater(tf.shape(p_ace05_net)[0], 0), 
                               lambda: p_ace05_loss, lambda: 0.0)
      else:
        p_ace05_loss = 0.0
      if params['augment_ere'] != 0.0:
        p_ere_loss, p_ere_prob = decoder(p_ere_net, p_ere_labels, mode, 
          params['ere_num_classes'], params, r_repr=ere_r_repr, loss_fn='ranking')
        p_ere_loss = tf.cond(tf.greater(tf.shape(p_ere_net)[0], 0), 
                            lambda: p_ere_loss, lambda: 0.0)
      else:
        p_ere_loss = 0.0
      augment_loss = p_ace05_loss * params['augment_ace05'] + p_ere_loss * params['augment_ere']
      tf.summary.scalar('augment_loss', augment_loss)
      return augment_loss
    else:
      return None


def add_correlation_constraint(ere2ace05_prob, ace052ere_prob, ere2ace05_index, ere2ace05_net, ace052ere_net, params):
  with tf.variable_scope('relation_type/meta'):
    ere2ace05_output = tf.get_variable('ere2ace05_prob', shape=ere2ace05_prob.shape, trainable=False)
    ace052ere_index = np.transpose(ere2ace05_index)
    nonzero_index = np.sum(ace052ere_index, axis=1)
    nonzero_index = list(set(np.nonzero(nonzero_index)[0]))
    ace052ere_index = ace052ere_index[nonzero_index]
    print(ace052ere_index)
    ace052ere_prob = tf.gather(ace052ere_prob, nonzero_index)
    ace052ere_net = tf.gather(ace052ere_net, nonzero_index)
    ace052ere_output = tf.get_variable('ace052ere_prob', shape=ace052ere_index.shape, trainable=False)
#    correlation_loss = correlation_constraint_loss(ere2ace05_prob[1:,1:], params['ere2ace05_index'][1:,1:])
                       # correlation_constraint_loss(ace052ere_prob, ace052ere_index)
    #correlation_loss = correlation_ranking_loss(ace052ere_net, ace052ere_index,
    #                                            params['rank_pos_margin'], params['rank_neg_margin'], params['rank_scale'])
    correlation_loss = (correlation_ranking_loss(ere2ace05_net, ere2ace05_index, 
                                                 params['rank_pos_margin'], params['rank_neg_margin'], params['rank_scale']))
    output_assign1 = tf.assign(ere2ace05_output, ere2ace05_prob)
    output_assign2 = tf.assign(ace052ere_output, ace052ere_prob)
  return correlation_loss, [output_assign1, output_assign2]

def relation_kshot_encoder(inputs, pretrained_embed, mode, nb_classes, params, reduce=True):
  embed = embed_layer(inputs, params['vocab_size'], params['max_len'],
                            params['entity_type_size'], params['word_embed_dim'],
                            params['pos_embed_dim'],
                            pretrained_embed, params['tune_word_embed'])
  hidden = bi_rnn_output(embed, params['hidden_rnn_size'])
  net = encoder(hidden, mode, params)
  net = fully_connected(inputs=net,
                        num_outputs=params['relation_embed_size'],
                        activation_fn=tf.nn.tanh, scope='output')
  print('kshot output', net)
  net = tf.reshape(net, [nb_classes, params['ksupport_size'], params['relation_embed_size']])
  if reduce:
    net = tf.reduce_mean(net, axis=1)# avg of embedding for the same type
#  with tf.variable_scope('negative', reuse=tf.AUTO_REUSE):
#    negative_base = tf.get_variable('negative', shape=[params['relation_embed_size']])
#  net = tf.concat([tf.expand_dims(net[0, :] + negative_base, axis=0), net[1:,:]], axis=0)
  return net

def kshot_to_normal(kshot_net, nb_classes, params):
  """convert kshot examples to normal examples with labels
     kshot_net: unreduced kshot encodings with shape 
         [nb_classes, params['ksupport_size'], params['relation_embed_size']]
  """
  print(kshot_net)
  kshot_net_by_label = tf.unstack(kshot_net, axis=0)
  net = []
  labels = []
  for i in range(len(kshot_net_by_label)):
    indices = tf.fill([kshot_net_by_label[i].shape[0]], i)
    labels += [tf.one_hot(indices, depth=nb_classes)]
    net += [kshot_net_by_label[i]]

  net = tf.concat(net, axis=0)
  labels = tf.concat(labels, axis=0)
  return net, labels

def kshot_decoder(net, labels, mode, num_classes, params, r_repr=None, loss_fn='sigmoid'):
  net = tf.expand_dims(net, axis=1) # label_dim
  r_repr = tf.expand_dims(r_repr, axis=0) # batch_dim,  nb_classes*kshot_size, embed_size
#  r_repr_normalized = tf.nn.l2_normalize(r_repr, 2)
#  net_normalized = tf.nn.l2_normalize(net, 2)
#  cosine  = tf.reduce_sum(tf.multiply(net_normalized, r_repr_normalized), axis=2)
  relation = -tf.norm(net - r_repr, ord=1, axis=2) # l1 distance
#  net = fully_connected(net, num_outputs=1, activation_fn=None, scope="convert_prob")
#  relation = tf.squeeze(net, axis=2)
#  relation = tf.reduce_sum(tf.multiply(net, r_repr), axis=2)
#  relation = tf.reshape(relation, [-1, num_classes, params['kshot_size']])
  
#  relation = tf.reduce_mean(relation, axis=2)
#  relation = tf.reshape(relation, [-1, num_classes, params['kshot_size']])
#  relation = tf.reduce_sum(relation, axis=2)
  relation_prob = tf.nn.softmax(relation)
  
  if mode != tf.estimator.ModeKeys.PREDICT:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
    loss = tf.reduce_mean(loss, 0) 
  else:
    loss = None
  return loss, relation_prob


def get_l1_distance_loss(r_mapping):
  return tf.reduce_mean(r_mapping)

def get_l1_distance(r_repr1, r_repr2):
  r_repr1 = tf.expand_dims(r_repr1, axis=1)
  r_repr2 = tf.expand_dims(r_repr2, axis=0)
  r = tf.abs(r_repr1 - r_repr2)
  print(r)
  r = tf.reduce_sum(r, axis=2) # l1 norm
  return r

def get_l2_distance(r_repr1, r_repr2):
  r_repr1 = tf.expand_dims(r_repr1, axis=1)
  r_repr2 = tf.expand_dims(r_repr2, axis=0)
  r = tf.square(r_repr1 - r_repr2)
  r = tf.reduce_sum(r, axis=2)
  r = tf.sqrt(r)
  return r

def get_ones_indices(matrix):
  index = tf.not_equal(matrix, 0.0)
  index = tf.where(index)
  return index

def add_transformation(ace05_r_repr, ere_r_repr, relation_embed_size, method='add'):
  init = tf.random_uniform_initializer(minval=-0.25, maxval=0.25)
  if method == 'add':
    ace05_matrix = tf.get_variable('ace05_matrix', shape=[relation_embed_size, relation_embed_size],
                                   trainable=True)
    ere_matrix = tf.get_variable('ere_matrix', shape=[relation_embed_size, relation_embed_size],
                                 trainable=True)
    ace05_r_repr = tf.matmul(ace05_r_repr, ace05_matrix)
    ere_r_repr = tf.matmul(ere_r_repr, ere_matrix)
  return ace05_r_repr, ere_r_repr


def repr_decoder(net, labels, mode, num_classes, params, r_repr=None, loss_fn='sigmoid'): #using sigmoid
  with tf.variable_scope("decoder"):
    net = fully_connected(inputs=net,
                               num_outputs=params['hidden_layer_dim'],
                               activation_fn=tf.nn.relu)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      net = tf.nn.dropout(net, 0.5)
    if r_repr != None:
      net = tf.tile(tf.expand_dims(net, axis=1), [1, r_repr.shape[0], 1])
      r_repr = tf.tile(tf.expand_dims(r_repr, axis=0),[tf.shape(net)[0], 1, 1]) # batch_dim
      net = tf.concat([net, r_repr], axis=2)
      relation = fully_connected(inputs=net,
                                 num_outputs=num_classes,
                                 activation_fn=None)
      relation = tf.linalg.diag_part(relation)
    else: # normal decoder
      relation = fully_connected(inputs=net,
                                 num_outputs=num_classes,
                                 activation_fn=None)
  if loss_fn == 'sigmoid':
    relation_prob = tf.nn.sigmoid(relation)
  else:
    relation_prob = tf.nn.softmax(relation)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return None, relation_prob
  if loss_fn == 'sigmoid':
    loss = tf.losses.sigmoid_cross_entropy(logits=relation, multi_class_labels=labels, reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
  else:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
    loss = tf.reduce_mean(loss, 0)

  return loss, relation_prob

def reader_decoder(net, labels, mode, num_classes, params, r_repr=None, loss_fn='sigmoid'):
  """
  Args:
    net: [batch_size, feature_size]
    r_repr: [num_classes, relation_embed_size]
    labels: [batch_size, num_classes]
  """
  relation, labels, weights = reader_decoder_helper(net, r_repr, labels, None, None, None, mode, params)
  print(relation)
  if loss_fn == 'sigmoid':
    relation_prob = tf.nn.sigmoid(relation)
  else:
    relation_prob = tf.nn.softmax(relation)
  if mode != tf.estimator.ModeKeys.PREDICT:
    if loss_fn == 'sigmoid':
      loss = tf.losses.sigmoid_cross_entropy(logits=relation, multi_class_labels=labels, reduction=tf.losses.Reduction.NONE)
      if not params['opt_other']:
        loss = loss[:, 1:]
      loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    else: # softmax for ere decoder
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
      loss = tf.reduce_mean(loss, 0)
  else:
    loss = None
  return loss, relation_prob, None

def label_embedding_decoder(net, labels, mode, num_classes, params, r_repr=None, loss_fn='sigmoid', weights=1.0):
  relation = get_label_embedding_logits(net, r_repr, params['embedding_dist'])
  print(relation)
  loss, relation_prob = get_loss_prob(relation, labels, mode, params, loss_fn, weights, params['opt_other'])
  return loss, relation_prob

def label_embedding_decoder_logits(net, labels, mode, num_classes, params, r_repr=None, loss_fn='sigmoid', weights=1.0):
#  net = tf.stop_gradient(net)
  relation = get_label_embedding_logits(net, r_repr, params['embedding_dist'])
  print(relation)
  loss, relation_prob = get_loss_prob(relation, labels, mode, params, loss_fn, weights, params['opt_other'])
  return loss, relation, relation_prob

def get_label_embedding_logits(net, r_repr, embedding_dist):
  net = tf.expand_dims(net, axis=1) # label_dim
  r_repr = tf.expand_dims(r_repr, axis=0) # batch_dim
  if embedding_dist == 'cosine':
    relation = tf.reduce_sum(tf.multiply(net, r_repr), axis=2)
  elif embedding_dist == 'cosine_norm':
    r_repr = tf.nn.l2_normalize(r_repr, 2)
    relation = tf.reduce_sum(tf.multiply(net, r_repr), axis=2)
  elif embedding_dist == 'l1':
    net = tf.abs(net - r_repr) # l1 distance
    net = fully_connected(net, num_outputs=1, activation_fn=None, scope="convert_prob")
    relation = tf.squeeze(net, axis=2)
  elif embedding_dist == 'l1_inv':
    net = tf.abs(net - r_repr) # l1 distance
    net = tf.reduce_sum(net, axis=2)
    relation = - net
  elif embedding_dist == 'l2':
    net = tf.square(net - r_repr)
    net = fully_connected(net, num_outputs=1, activation_fn=None, scope="convert_prob")
    relation = tf.squeeze(net, axis=2)
  elif embedding_dist == 'l2_inv':
    net = tf.square(net - r_repr) # l1 distance
    net = tf.reduce_sum(net, axis=2)
    net = tf.sqrt(net)
    relation = - net  
  return relation

def get_loss_prob(relation, labels, mode, params, loss_fn='sigmoid', weights=1.0, opt_other=True):
  if loss_fn == 'sigmoid':
    relation_prob = tf.nn.sigmoid(relation)
  else: # softmax
    relation_prob = tf.nn.softmax(relation)

  if mode == tf.estimator.ModeKeys.TRAIN:
    if loss_fn == 'sigmoid':
      loss = tf.losses.sigmoid_cross_entropy(logits=relation, multi_class_labels=labels, reduction=tf.losses.Reduction.NONE)
      if not opt_other:
        loss = loss[:, 1:]
      loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    elif loss_fn == 'ranking': # softmax  for ere decoder
      loss = ranking_loss(relation, labels, params['rank_pos_margin'], params['rank_neg_margin'], params['rank_scale'])
    else: # softmax
      loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=relation, weights=weights, reduction=tf.losses.Reduction.NONE)
#      loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
      loss = tf.reduce_mean(loss, 0)
  else:
    loss = None
  return loss, relation_prob

def joint_separate_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None
  if params['pretrain_w2v']:
    pretrained_embed = tf.to_float(tf.constant(params['embed']))
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder"):
    with tf.variable_scope("shared") as scope:
      ace05_embed = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                              params['entity_type_size'], params['word_embed_dim'],
                              params['pos_embed_dim'],
                              pretrained_embed, params['tune_word_embed'])
      ace05_shared = encoder(ace05_embed, mode, params)
      scope.reuse_variables()
      ere_embed = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
      ere_shared = encoder(ere_embed, mode, params)

    with tf.variable_scope('ace05'):
      ace05_net = encoder(ace05_embed, mode, params)

    with tf.variable_scope('ere'):
      ere_net = encoder(ere_embed, mode, params)

    with tf.variable_scope('ace05'):
      ace05_net = tf.concat([ace05_net, ace05_shared], axis=1)
      ace05_prob, ace05_hidden, ace05_loss = decoder_hidden(ace05_net, ace05_labels, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
    with tf.variable_scope('ere'):
      ere_net = tf.concat([ere_net, ere_shared], axis=1)
      ere_prob, ere_hidden, ere_loss = decoder_hidden(ere_net, ere_labels, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)


  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, None, mode, params)


def joint_relation_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None
  if params['pretrain_w2v']:
    pretrained_embed = tf.to_float(tf.constant(params['embed']))
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as scope:
    ace05_net = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    ace05_net = encoder(ace05_net, mode, params)
    if params['encoder_hidden_layer']:
      ace05_net = hidden_layer(ace05_net, 1, 300, mode)
    scope.reuse_variables()
    ere_net = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    ere_net = encoder(ere_net, mode, params)
    if params['encoder_hidden_layer']:
      ere_net = hidden_layer(ere_net, 1, 300, mode)
  if params['shared_hidden']:
    ace05_prob, ere_prob, ace05_loss, ere_loss, ace05_shared, ere_shared = shared_hidden_decoder(
        ace05_net, ace05_labels, params['ace05_num_classes'],
        ere_net, ere_labels, params['ere_num_classes'],
        params['shared_hidden_dim'], 
        params['hidden_layer_dim'] - params['shared_hidden_dim']/2,
        mode)
    if params['da']:
      if params['da_input'] == 'feature':
        ace05_da_input, ere_da_input = ace05_net, ere_net
      elif params['da_input'] == 'hidden':
        ace05_da_input, ere_da_input = ace05_shared, ere_shared
      elif params['da_input'] == 'both':
        ace05_da_input = tf.concat([ace05_net, ace05_shared], axis=1)
        ere_da_input = tf.concat([ere_net,  ere_shared], axis=1)
      domain_loss = add_adversarial_loss(ace05_da_input, ere_da_input)
    else:
      domain_loss = None
    adv_loss = None
  else: # completely separate decoders
    with tf.variable_scope('ace05'):
      ace05_prob, ace05_hidden, ace05_loss = decoder_hidden(ace05_net, ace05_labels, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
    with tf.variable_scope('ere'):
      ere_prob, ere_hidden, ere_loss = decoder_hidden(ere_net, ere_labels, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
    if params['da'] and params['da_input'] == 'hidden':
      with tf.variable_scope('ace05', reuse=True):
        ere_prob_by_ace05, ere_hidden_by_ace05, _ = decoder_hidden(ere_net, None, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], tf.estimator.ModeKeys.PREDICT)
      with tf.variable_scope('ere', reuse=True):
        ace05_prob_by_ere, ace05_hidden_by_ere, _ = decoder_hidden(ace05_net, None, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], tf.estimator.ModeKeys.PREDICT)
      ace05_adv_loss = add_adversarial_loss(ace05_hidden, ace05_hidden_by_ere, 'ace05')
      ere_adv_loss = add_adversarial_loss(ere_hidden, ere_hidden_by_ace05, 'ere')
      adv_loss = ace05_adv_loss + ere_adv_loss
    elif params['da'] and params['da_input'] == 'feature':
      adv_loss = add_adversarial_loss(ace05_net, ere_net)
    else:
      adv_loss = None
  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, adv_loss, mode, params) 

def joint_auxiliary_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None
  if params['pretrain_w2v']:
    pretrained_embed = tf.to_float(tf.constant(params['embed']))
  else:
    pretrained_embed = None
  with tf.variable_scope('base_model'):
    with tf.variable_scope('encoder') as scope:
      ace05_net = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ace05_net = encoder(ace05_net, mode, params)
      scope.reuse_variables()
      ere_net = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                        params['entity_type_size'], params['word_embed_dim'],
                        params['pos_embed_dim'],
                        pretrained_embed, params['tune_word_embed'])
      ere_net = encoder(ere_net, mode, params)
    with tf.variable_scope('ace05') as scope:
      ace05_prob, ace05_loss = decoder(ace05_net, ace05_labels, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
      scope.reuse_variables()
      ere_by_ace05_prob, _ = decoder(ere_net, None, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], tf.estimator.ModeKeys.PREDICT)
    with tf.variable_scope('ere') as scope:
      ere_prob, ere_loss = decoder(ere_net, ere_labels, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
      scope.reuse_variables()
      ace05_by_ere_prob, _ = decoder(ace05_net, None, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], tf.estimator.ModeKeys.PREDICT)
    if params['da']:
      domain_loss = add_adversarial_loss(ace05_net, ere_net)
    else:
      domain_loss = None
    ace05_net = tf.concat([ace05_prob, ace05_by_ere_prob], axis=1)
    ere_net = tf.concat([ere_prob, ere_by_ace05_prob], axis=1)
  with tf.variable_scope('ensemble'):
    with tf.variable_scope('ace05_2'):
      ace05_prob2, ace05_loss2 = decoder(ace05_net, ace05_labels, params['ace05_num_classes'], 0, 0, mode)
    with tf.variable_scope('ere_2'):
      ere_prob2, ere_loss2 = decoder(ere_net, ere_labels, params['ere_num_classes'], 0, 0, mode)
  if params['pretrained_model'] is not None and params['current_epoch'] == 1 and mode == tf.estimator.ModeKeys.TRAIN:
    tf.contrib.framework.init_from_checkpoint(params['pretrained_model'], {'/': 'base_model/'})
  variables = tf.trainable_variables('ensemble')
  return make_ace05_ere_spec(ace05_prob2, ace05_loss2, ere_prob2, ere_loss2, domain_loss, mode, params, variables) 


def joint_transfer_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None
  if params['pretrain_w2v']:
    pretrained_embed = tf.to_float(tf.constant(params['embed']))
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as scope:
    ace05_net = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    ace05_net = encoder(ace05_net, mode, params)
    scope.reuse_variables()
    ere_net = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    ere_net = encoder(ere_net, mode, params)

  if params['joint'] == 'joint_ere2ace':
    ace05_prob, ere_prob, ace05_loss, ere_loss = transfer_decoder(
        ace05_net,ace05_labels, params['ace05_num_classes'], 'ace05',
        ere_net, ere_labels, params['ere_num_classes'], 'ere',
        params['num_hidden_layer'], params['hidden_layer_dim'], params['joint_decoder'], mode)
  elif params['joint'] == 'joint_ace2ere':
    ere_prob, ace05_prob, ere_loss, ace05_loss = transfer_decoder(
        ere_net, ere_labels, params['ere_num_classes'], 'ere',
        ace05_net,ace05_labels, params['ace05_num_classes'], 'ace05',
        params['num_hidden_layer'], params['hidden_layer_dim'], params['joint_decoder'], mode)
  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, None, mode, params) 

def joint_meta_model(features, labels, mode, params):
  ace05_inputs = features['ace05_word'], features['ace05_type'], features['ace05_pos1'], features['ace05_pos2']
  ere_inputs = features['ere_word'], features['ere_type'], features['ere_pos1'], features['ere_pos2']
  if mode == tf.estimator.ModeKeys.TRAIN:
    ace05_labels = labels['ace05_relation']
    ere_labels = labels['ere_relation']
  else:
    ace05_labels, ere_labels = None, None
  if params['pretrain_w2v']:
    pretrained_embed = tf.to_float(tf.constant(params['embed']))
  else:
    pretrained_embed = None
  with tf.variable_scope("encoder") as scope:
    ace05_net = embed_layer(ace05_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    ace05_net = encoder(ace05_net, mode, params)
    if params['encoder_hidden_layer']:
      ace05_net = hidden_layer(ace05_net, 1, 300, mode)
    scope.reuse_variables()
    ere_net = embed_layer(ere_inputs, params['vocab_size'], params['max_len'],
                      params['entity_type_size'], params['word_embed_dim'],
                      params['pos_embed_dim'],
                      pretrained_embed, params['tune_word_embed'])
    ere_net = encoder(ere_net, mode, params)
    if params['encoder_hidden_layer']:
      ere_net = hidden_layer(ere_net, 1, 300, mode)

  with tf.variable_scope('ace05') as scope:
    ace05_prob, ace05_hidden, ace05_loss = decoder_hidden(ace05_net, ace05_labels, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
    scope.reuse_variables()
    ere_prob_by_ace05, ere_hidden_by_ace05, _ = decoder_hidden(ere_net, None, params['ace05_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], tf.estimator.ModeKeys.PREDICT)
  with tf.variable_scope('ere') as scope:
    ere_prob, ere_hidden, ere_loss = decoder_hidden(ere_net, ere_labels, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], mode)
    scope.reuse_variables()
    ace05_prob_by_ere, ace05_hidden_by_ere, _ = decoder_hidden(ace05_net, None, params['ere_num_classes'], params['num_hidden_layer'], params['hidden_layer_dim'], tf.estimator.ModeKeys.PREDICT)
  if mode == tf.estimator.ModeKeys.TRAIN:
    print(params['meta_input'])  
    with tf.variable_scope('meta') as scope:
      ace05_meta_labels = tf.argmax(ace05_labels, axis=1)
      ere_meta_labels = tf.argmax(ere_labels, axis=1)
      if params['meta_input'] == 'output':
        ace05_meta_loss = meta_classifier(ace05_prob_by_ere, ace05_meta_labels, 'ace05')
        ere_meta_loss = meta_classifier(ere_prob_by_ace05, ere_meta_labels, 'ere')
        meta_loss = ace05_meta_loss + ere_meta_loss
      elif params['meta_input'] == 'feature':
        meta_net = tf.concat([ace05_net, ere_net], axis=0)
        meta_labels = tf.concat([ace05_meta_labels, ere_meta_labels], axis=0)
        meta_loss = meta_classifier(meta_net, meta_labels)
      elif params['meta_input'] == 'hidden':
        ace05_meta_loss = meta_classifier(ace05_hidden_by_ere, ace05_meta_labels, 'ace05')
        ere_meta_loss = meta_classifier(ere_hidden_by_ace05, ere_meta_labels, 'ere')
        meta_loss = ace05_meta_loss + ere_meta_loss
      else:
        raise NotImplementedError
  else:
    meta_loss = None
  return make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, None, mode, params, meta_loss=meta_loss)  

def meta_classifier(net, labels, name): # is_relatoin?
  """
  Args:
    labels: indices of relations, 0 as other, nonzero as relations
  """
  labels = tf.cast(labels, tf.bool) # non-zero to True
  labels = tf.cast(labels, tf.int64) # to 1, 0
  one_hot_labels = tf.one_hot(labels, depth=2)
#  net = fully_connected(net, 100, activation_fn=tf.nn.relu) # add one hidden
  is_relation = fully_connected(net, 2, activation_fn=None)
  losses = tf.nn.softmax_cross_entropy_with_logits(logits=is_relation, labels=one_hot_labels)
  loss = tf.reduce_mean(losses, 0)
  is_relation = tf.argmax(is_relation, axis=1)
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(labels, is_relation)))
  tf.summary.scalar('meta_classifier_loss/'+name, loss)
  tf.summary.scalar('meta_classifier_accuracy/'+name, accuracy)
  return loss

def add_adversarial_loss(ace05_net, ere_net, name=""):
  """this can be used for more general cases other than ace05 vs ere"""
  ace05_domain = tf.zeros([tf.shape(ace05_net)[0]], dtype=tf.int32)
  ace05_domain = tf.one_hot(ace05_domain, depth=2)
  ere_domain = tf.ones([tf.shape(ere_net)[0]], dtype=tf.int32)
  ere_domain = tf.one_hot(ere_domain, depth=2)
  whole_repr = tf.concat([ace05_net, ere_net], axis=0)
  domain = tf.concat([ace05_domain, ere_domain], axis=0)
  domain_loss = domain_adversarial(whole_repr, domain, 2, name)
  return domain_loss

def add_l2_loss(ace05_net, ere_net, name=""):
  """based on the avg of two datasets, computed as a batch"""
  ace05_avg = tf.reduce_mean(ace05_net, axis=0)
  ere_avg = tf.reduce_mean(ere_net, axis=0)
  return tf.nn.l2_loss(ace05_avg-ere_avg) # sum(t ** 2) / 2


def domain_adversarial(net, domain_label, num_domains, name=""):
  with tf.variable_scope('adversarial/'+name):
    net = 2*tf.stop_gradient(net)-net
    net = fully_connected(net, num_domains, activation_fn=None)
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=domain_label)
    domain_loss = tf.reduce_mean(losses, 0)
    net = tf.argmax(net, axis=1)
    domain_label = tf.argmax(domain_label, axis=1)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(domain_label, net)))
    tf.summary.scalar('adversarial_loss/'+name, domain_loss)
    tf.summary.scalar('adversarial_accuracy/'+name, accuracy)
  return domain_loss

def transfer_decoder(net1, labels1, num_classes1, net1_name,
                     net2, labels2, num_classes2, net2_name,
                     num_layer, hidden_layer_dim, decoder_mode, train_mode):
  """net1 as main task, net2 as auxiliary"""
  with tf.variable_scope('decoder'):
    with tf.variable_scope(net2_name) as scope:
      net2_relation, net2_hidden = get_prob(net2, num_classes2, num_layer, hidden_layer_dim, train_mode)
      scope.reuse_variables()
      net1_relation_by_net2, net1_hidden_by_net2 = get_prob(net1, num_classes2, num_layer, hidden_layer_dim, train_mode)
    with tf.variable_scope(net1_name):
      if decoder_mode == 'cascade':
        net1 = tf.concat([net1, net1_hidden_by_net2], axis=1)
      net1_hidden = hidden_layer(net1, num_layer, hidden_layer_dim, train_mode)
      if decoder_mode == 'forward':
        net1_relation_by_net2 = tf.stop_gradient(net1_relation_by_net2)
      net1_hidden = tf.concat([net1_hidden, net1_relation_by_net2], axis=1)
      net1_relation = fully_connected(inputs=net1_hidden,
                                      num_outputs=num_classes1,
                                      activation_fn=None,
                                      scope='fc_output')
    net1_prob = tf.nn.softmax(net1_relation)
    net2_prob = tf.nn.softmax(net2_relation)
    if train_mode == tf.estimator.ModeKeys.PREDICT:
      return net1_prob, net2_prob, None, None
    else:
      loss1 = get_loss(net1_relation, labels1, train_mode)
      loss2 = get_loss(net2_relation, labels2, train_mode)
      return net1_prob, net2_prob, loss1, loss2



def get_loss(logits, labels, mode):
  if mode == tf.estimator.ModeKeys.PREDICT:
    loss = None
  else:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss, 0)
  return loss

def hidden_layer(net, num_layer, hidden_layer_dim, mode):
  for i in range(num_layer):
    net = fully_connected(inputs=net,
                           num_outputs=hidden_layer_dim,
                           activation_fn=tf.nn.relu,
                           scope='fc_hidden'+str(i))
    if mode == tf.estimator.ModeKeys.TRAIN:
        net = tf.nn.dropout(net, 0.5)
  return net

def get_prob(net, num_classes, num_layer, hidden_layer_dim, mode):
  net = hidden_layer(net, num_layer, hidden_layer_dim, mode)
  relation = fully_connected(inputs=net,
                             num_outputs=num_classes,
                             activation_fn=None,
                             scope='fc_output')
  return relation, net

def shared_hidden_decoder(ace05_net, ace05_labels, ace05_num_classes,
               ere_net, ere_labels, ere_num_classes,
               shared_hidden_dim, task_specific_dim, mode):
  with tf.variable_scope('decoder'):
    ace05_hidden = fully_connected(ace05_net, num_outputs=task_specific_dim, activation_fn=tf.nn.relu)
    ere_hidden = fully_connected(ere_net, num_outputs=task_specific_dim, activation_fn=tf.nn.relu)
    with tf.variable_scope('shared_hidden') as scope:
      ace05_shared = fully_connected(ace05_net, num_outputs=shared_hidden_dim, activation_fn=tf.nn.relu, scope='fully_connected')
      scope.reuse_variables()
      ere_shared = fully_connected(ere_net, num_outputs=shared_hidden_dim, activation_fn=tf.nn.relu, scope='fully_connected')
    ace05_hidden = tf.concat([ace05_hidden, ace05_shared], axis=1)
    ere_hidden = tf.concat([ere_hidden, ere_shared], axis=1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        ace05_hidden = tf.nn.dropout(ace05_hidden, 0.5)
        ere_hidden = tf.nn.dropout(ere_hidden, 0.5)
    ace05_relation = fully_connected(inputs=ace05_hidden,
                             num_outputs=ace05_num_classes,
                             activation_fn=None)
    ere_relation = fully_connected(inputs=ere_hidden,
                             num_outputs=ere_num_classes,
                             activation_fn=None)
    ace05_loss = get_loss(ace05_relation, ace05_labels, mode)
    ere_loss = get_loss(ere_relation, ere_labels, mode)
    return ace05_relation, ere_relation, ace05_loss, ere_loss, ace05_shared, ere_shared

def decoder(net, labels, num_classes, num_layer, hidden_layer_dim, mode):
  with tf.variable_scope('decoder'):
    for i in range(num_layer):
      net = fully_connected(inputs=net,
                            num_outputs=hidden_layer_dim,
                            activation_fn=tf.nn.relu)
      if mode == tf.estimator.ModeKeys.TRAIN:
        net = tf.nn.dropout(net, 0.5)
    relation = fully_connected(inputs=net,
                               num_outputs=num_classes,
                               activation_fn=None)
  if mode == tf.estimator.ModeKeys.PREDICT:
    loss = None
  else:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
    loss = tf.reduce_mean(loss, 0)
  return relation, loss

def decoder_hidden(net, labels, num_classes, num_layer, hidden_layer_dim, mode):
  with tf.variable_scope('decoder'):
    for i in range(num_layer):
      net = fully_connected(inputs=net,
                            num_outputs=hidden_layer_dim,
                            activation_fn=tf.nn.relu)
      if mode == tf.estimator.ModeKeys.TRAIN:
        net = tf.nn.dropout(net, 0.5)
    relation = fully_connected(inputs=net,
                               num_outputs=num_classes,
                               activation_fn=None)
  if mode == tf.estimator.ModeKeys.PREDICT:
    loss = None
  else:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=relation, labels=labels)
    loss = tf.reduce_mean(loss, 0)
  return relation, net, loss
  
def make_estimator_spec(relation_prob, loss, mode, params, variables=None):
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"relation": relation_prob})

  if params['lr_decay']:
    learning_rate_decay_fn = partial(tf.train.exponential_decay,
                                   decay_steps=params['learning_rate_decay_step'],
                                   decay_rate=0.5, staircase=True)
  else:
    learning_rate_decay_fn = None

  train_op = layers.optimize_loss(
      loss, tf.train.get_global_step(),
      variables=variables,
      learning_rate_decay_fn=learning_rate_decay_fn,
      learning_rate=params['learning_rate'],
      optimizer='Adam')

  if mode == tf.estimator.ModeKeys.TRAIN:
    print(utils.variable_shapes())

  return tf.estimator.EstimatorSpec(mode, {"relation": relation_prob}, loss, train_op)

def make_ace05_ere_spec(ace05_prob, ace05_loss, ere_prob, ere_loss, domain_loss, mode, params, meta_loss=None, variables=None, 
                        similarity_loss=None, augment_loss=None):
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'ace05_relation': ace05_prob, 'ere_relation': ere_prob})

  if params['lr_decay']:
    learning_rate_decay_fn = partial(tf.train.exponential_decay,
                                   decay_steps=params['learning_rate_decay_step'],
                                   decay_rate=params['lr_decay_rate'], staircase=params['lr_staircase'])
  else:
    learning_rate_decay_fn = None
  alpha1 = 1.0-params['lamda']
  alpha2 = params['lamda']
  loss = ace05_loss * alpha1 + ere_loss * alpha2
  if similarity_loss is not None:
    loss += similarity_loss * params['beta']
    tf.summary.scalar('similarity_loss', similarity_loss)
  if augment_loss is not None:
    loss += augment_loss * params['augment']
  if domain_loss is not None:
    loss += domain_loss * params['alpha3']
  if meta_loss is not None:
    alpha_meta = params['alpha_meta']
#    alpha_meta = tf.train.exponential_decay(params['alpha_meta'], 
#       tf.train.get_global_step(), decay_steps=680*2, decay_rate=0.5, staircase=True)
#    tf.summary.scalar('alpha_meta', alpha_meta)
    loss += meta_loss * alpha_meta

  if params['constraint_loss'] != 0.0:
    constraint_losses = tf.get_collection('r_repr_constraint')
    constraint_loss = 0.0
    for l in constraint_losses:
      constraint_loss += l
    tf.summary.scalar('constraint_loss', constraint_loss)
    loss += constraint_loss*params['constraint_loss']
  tf.summary.scalar('ace05_loss', ace05_loss)
  tf.summary.scalar('ere_loss', ere_loss) 
  train_op = layers.optimize_loss(
      loss, tf.train.get_global_step(),
      variables=variables,
      learning_rate_decay_fn=learning_rate_decay_fn,
      learning_rate=params['learning_rate'],
      optimizer='Adam')

  if mode == tf.estimator.ModeKeys.TRAIN:
    print(utils.variable_shapes())

  return tf.estimator.EstimatorSpec(mode, {'ace05_relation': ace05_prob, 'ere_relation': ere_prob}, loss, train_op)
