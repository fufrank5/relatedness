from __future__ import print_function
import cPickle
import numpy as np
from collections import Counter

RELATION_FEATURE_FILE = './data/ace05/relation_dep.pkl'
DATA_PATH = './data/ace05/'


CONVERT_UNDIRECT = {
  'NONE': 'NONE',
  'GEN-AFF(e1,e2)': 'GEN-AFF',
  'GEN-AFF(e2,e1)': 'GEN-AFF',
  'PART-WHOLE(e1,e2)': 'PART-WHOLE',
  'PART-WHOLE(e2,e1)': 'PART-WHOLE',
  'PER-SOC': 'PER-SOC',
  'PHYS': 'PHYS',
  'ART(e1,e2)': 'ART',
  'ART(e2,e1)': 'ART',
  'ORG-AFF(e1,e2)': 'ORG-AFF',
  'ORG-AFF(e2,e1)': 'ORG-AFF',
}


def load_dataset(relation_file=RELATION_FEATURE_FILE, max_len=50, directional=True):
  with open(relation_file, 'rb') as input:
    relations, vocab, entity_type, dep_label, label_set, embedding = cPickle.load(input)
  embed = embedding['word_embed']
  vocab, embed = normalize_vocab_embed(vocab, embed)
  direct_inverse = {v:k for k,v in label_set.iteritems()}
  if not directional:
    label_set = {'NONE': 0,  'GEN-AFF': 1, 'PART-WHOLE':2, 'PER-SOC':3, 'PHYS':4, 'ART': 5, 'ORG-AFF':6}
  label_names, names_vocab, names_embed = process_label_names(label_set, vocab, embed, directional=directional)
  print(label_names, names_vocab)
  print(embed.shape)
  print(dep_label, len(dep_label))
  print(label_set, entity_type, len(embedding['type_embed']), len(embedding['pos1_embed']))
  
  feature_keys = ('word', 'type', 'pos1', 'pos2', 'dep', 'dep_path', 
                  'has_dep', 'on_dep_path', 'chunk')
  trainX = {k : [] for k in feature_keys}
  devX = {k : [] for k in feature_keys}
  trainY, devY = [], []
  test = {'bc1': None, 'cts': None, 'wl': None}
  for corpus in test:
    test[corpus] = ({k : [] for k in feature_keys}, [])
  print(relations[0])
  for rel in relations:
    if rel['corpus'] == 'bn_nw':
      features = process_features(rel, vocab, dep_label, max_len)
      for feature in feature_keys:
        trainX[feature] += [features[feature]]
      trainY += [rel['y'] if directional else label_set[CONVERT_UNDIRECT[direct_inverse[rel['y']]]]]
    elif rel['corpus'] == 'bc0':
      features = process_features(rel, vocab, dep_label, max_len)
      for feature in feature_keys:
        devX[feature] += [features[feature]]
      devY += [rel['y'] if directional else label_set[CONVERT_UNDIRECT[direct_inverse[rel['y']]]]]
    elif rel['corpus'] in ('bc1', 'cts', 'wl'):
      features = process_features(rel, vocab, dep_label, max_len)
      for feature in feature_keys:
        test[rel['corpus']][0][feature] += [features[feature]]
      test[rel['corpus']][1].append(rel['y'] if directional else label_set[CONVERT_UNDIRECT[direct_inverse[rel['y']]]])
  print('ace05 data loaded')
  label_index = {v: k for k, v in label_set.iteritems()}
  print('train (bn_nw):', len(trainY), count_labels(trainY, label_index))
  print('dev (bc0):', len(devY), count_labels(devY, label_index))
  print('test (bc1):', len(test['bc1'][1]), count_labels(test['bc1'][1], label_index))
  print('test (cts):', len(test['cts'][1]), count_labels(test['cts'][1], label_index))
  print('test (wl):', len(test['wl'][1]), count_labels(test['wl'][1], label_index))
  return ((trainX, trainY), (devX, devY), test), embed


def process_label_names(label_vocab, vocab, word_embed, subtype=False, extended_tag=False, append_tag=False, directional=True, embed_size=None):
  final_list = [0]* len(label_vocab)
  label_name_vocab = {'</s>':0, '<s>':1, '<ACE05>': 2, '<right>':3, '<left>':4,
                     # '<ACE05_PART-WHOLE>':5, '<ACE05_ORG-AFF>':6, '<ACE05_GEN-AFF>': 7,'<ACE05_PER-SOC>':8, '<ACE05_PHYS>':9 
                     }
  def random_vector(size):
    return np.random.uniform(low=-0.25, high=0.25, size=[size])
  if word_embed is None:
    label_name_embed = [random_vector(len(word_embed[0])), random_vector(len(word_embed[0])), random_vector(len(word_embed[0])),
                       random_vector(len(word_embed[0])), random_vector(len(word_embed[0]))]
  else:                  
    label_name_embed = [word_embed[vocab['<s>']], word_embed[vocab['</s>']],
                        random_vector(len(word_embed[0])), random_vector(len(word_embed[0])), random_vector(len(word_embed[0]))]
  if extended_tag:
    names_split = {
      'NONE': ['/', 'other'],
      'PART-WHOLE(e1,e2)': ['/', 'part', 'whole', '<ACE05_PART-WHOLE>', '<right>'],
      'PART-WHOLE(e2,e1)': ['/', 'part', 'whole', '<ACE05_PART-WHOLE>', '<left>'],
      'ORG-AFF(e1,e2)': ['/', 'organization', 'affiliation', '<ACE05_ORG-AFF>', '<right>'],
      'ORG-AFF(e2,e1)': ['/', 'organization', 'affiliation', '<ACE05_ORG-AFF>', '<left>'],
      'GEN-AFF(e1,e2)': ['/', 'general', 'affiliation', '<ACE05_GEN-AFF>', '<right>'],
      'GEN-AFF(e2,e1)': ['/', 'general', 'affiliation', '<ACE05_GEN-AFF>', '<left>'],
      'ART(e1,e2)': ['/', 'artifact', '<right>'],
      'ART(e2,e1)': ['/', 'artifact', '<left>'],
      'PER-SOC': ['/', 'personal', 'social', '<ACE05_PER-SOC>'],
      'PHYS': ['/', 'physical', '<ACE05_PHYS>'],
    }
  else:
    names_split = {
      'NONE': ['/', 'other'],
      'PART-WHOLE(e1,e2)': ['/', 'part', 'whole', '<right>'],
      'PART-WHOLE(e2,e1)': ['/', 'part', 'whole', '<left>'],
      'ORG-AFF(e1,e2)': ['/', 'organization', 'affiliation', '<right>'],
      'ORG-AFF(e2,e1)': ['/', 'organization', 'affiliation', '<left>'],
      'GEN-AFF(e1,e2)': ['/', 'general', 'affiliation', '<right>'],
      'GEN-AFF(e2,e1)': ['/', 'general', 'affiliation', '<left>'],
      'ART(e1,e2)': ['/', 'artifact', '<right>'],
      'ART(e2,e1)': ['/', 'artifact', '<left>'],
      'PER-SOC': ['/', 'personal', 'social'],
      'PHYS': ['/', 'physical'],
    }
  if directional is False:
    names_split = {
      'NONE': ['/', 'other'],
      'PART-WHOLE': ['/', 'part', 'whole'],
      'ORG-AFF': ['/', 'organization', 'affiliation'],
      'GEN-AFF': ['/', 'general', 'affiliation'],
      'ART': ['/', 'artifact'],
      'PER-SOC': ['/', 'personal', 'social'],
      'PHYS': ['/', 'physical'],
    }
  for key in names_split:
    if append_tag:
      names_split[key] += ['<ACE05>']
  for name, index in label_vocab.iteritems():
    name_desc = []
    for token in names_split[name]:
      if token == '/':
        continue
      #token = '<s>'

      if token not in label_name_vocab:
        label_name_vocab[token] = len(label_name_vocab)
        if word_embed is None or token not in vocab:
          label_name_embed += [random_vector(len(word_embed[0]))]
        else: 
          label_name_embed += [word_embed[vocab[token]]]
      name_desc += [label_name_vocab[token]]
    final_list[index] = name_desc
  label_name_embed = np.array(label_name_embed, dtype=np.float32)
  print(len(label_name_vocab))
  print(label_name_embed.shape)
  return final_list, label_name_vocab, label_name_embed

def count_labels(Y, label_index):
  c = Counter(Y)
  dist = []
  index_count = c.most_common()
  for i, c in index_count:
    dist += [(label_index[i], c)]
  return dist

def normalize_vocab_embed(vocab, embed):
  # add several defaults to be consistent with other dataset
  normalized_vocab = {'</s>': 0, '<UNK>':1, '<s>':2}
  normalized_embed = np.random.uniform(low=-0.25, high=0.25, size=[3, 300])
  for word, index in vocab.items():
    normalized_vocab[word] = vocab[word] + 3
  normalized_embed = np.concatenate((normalized_embed, embed), axis=0)
  return normalized_vocab, normalized_embed

def read_entity_vocab(relation_file=RELATION_FEATURE_FILE, directional=True):
  with open(relation_file, 'rb') as input:
    relations, vocab, entity_type, dep_label, label_set, embedding = cPickle.load(input)
  if not directional:
    label_set = {'NONE': 0,  'GEN-AFF': 1, 'PART-WHOLE':2, 'PER-SOC':3, 'PHYS':4, 'ART': 5, 'ORG-AFF':6}
  embed = embedding['word_embed']
  vocab, embed = normalize_vocab_embed(vocab, embed)
  return entity_type, vocab, label_set

def process_features(rel, vocab, dep_index, max_len):
  text = rel['text']
  tokens = text.split()
  features = {}
  features['word'] = textToId(tokens, vocab)
  features['type'] = textToType(tokens, rel)
  features['pos1'], features['pos2'] = textToPos(tokens, rel, max_len)
  features['has_dep'] = rel['grammar']
  features['on_dep_path'] = on_dep_path(rel['depIdx'], len(tokens))
  features['chunk'] = rel['prepreter']
  dep_tokens = rel['depRel'].split()
  features['dep_path'] = rel['depRel'].replace(' ', '_')
  features['dep'] = dep_path(dep_tokens, dep_index)
  return features

def on_dep_path(depIdx, sent_len):
  v = [0] * sent_len
  for i in depIdx:
    v[i] = 1
  return v

def dep_path(deps, dep_index):
  path = []
  for label in deps:
    if label in dep_index:
      index = dep_index[label] + 45 # nn
    else:
      index = -dep_index[label[:-1]] + 46 # nn'
    path += [index]
  return path

def textToType(tokens, rel):
  type_id = []
  for i in xrange(len(tokens)):
    if i == rel['pos1']:
      type_id += [rel['type1']]
    elif i == rel['pos2']:
      type_id += [rel['type2']]
    else:
      type_id += [1] # other is 0?
  return type_id

def textToPos(tokens, rel, max_len=50):
  pos1_id = []
  pos2_id = []
  for i in xrange(len(tokens)):
    pos1 = i - rel['pos1'] + max_len
    if pos1 < 0:
      pos1 = 0
    if pos1 >= max_len*2:
      pos1 = max_len * 2 - 1
    pos1_id += [pos1]
    pos2 = i - rel['pos2'] + max_len
    if pos2 < 0:
      pos2 = 0
    if pos2 >= max_len*2:
      pos2 = max_len * 2 - 1
    pos2_id += [pos2]
  
  return pos1_id, pos2_id      

def textToId(tokens, vocab):
  ids = []
  unknown_token = set()
  for token in tokens:
    try:
      token_id = vocab[token]
      ids += [token_id]
    except:
      ids += [1]
      unknown_token.add(token)
#  if len(unknown_token) != 0:
#    print 'unkown:', unknown_token # '_' ingored TODO
#    print tokens
  return ids

          
if __name__ == '__main__':
  ((trainX, trainY), (devX, devY), test), embed = load_dataset('./data/ace05/relation_dep.pkl', directional=True)
  entity_set, vocab, label_set = read_entity_vocab()
  normalize_vocab_embed(vocab, embed)
