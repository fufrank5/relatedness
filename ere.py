from __future__ import print_function

import random
import utils
from collections import Counter
import numpy as np
DATA_PATH = './data/ERE/ere_directional.txt'
DOC_ID_PATH = './data/ERE/'
WORD2VEC_FILE = './data//word2vec/GoogleNews-vectors-negative300.bin'

PREDEFINED_LABEL_SET = {
  'OTHER': 0, 
  'generalaffiliation(e1,e2)': 1,
  'generalaffiliation(e2,e1)': 8, 
  'physical': 4, 
  'orgaffiliation(e2,e1)': 5,
  'orgaffiliation(e1,e2)': 6,
  'partwhole(e1,e2)': 2,
  'partwhole(e2,e1)': 7,
  'personalsocial': 3}

def read_relation_candidates(relation_file):
  relations = []
  with open(relation_file) as f:
    ln = 1
    relation = {}
    for line in f:
       if ln % 6 == 0:  # empty line
         relations += [relation]
         relation = {}
       else:
         line = line.strip().split('\t')
         if ln % 6 == 5:  # label
           relation[line[1]] = line[2]
         else:
           relation[line[1]] = line[2].split()
         if 'doc_id' not in relation:
           relation['doc_id'] = line[0]
       ln += 1
  return relations

def write_relation_candidates(relations, relation_file):
  with open(relation_file, 'w') as f:
    for r in relations:
      print(r['doc_id'] + '\ttoken\t' + ' '.join(r['token']), file=f)
      print(r['doc_id'] + '\tentity_type\t' + ' '.join(r['entity_type']), file=f)
      print(r['doc_id'] + '\targ1\t' + ' '.join(r['arg1']), file=f)
      print(r['doc_id'] + '\targ2\t' + ' '.join(r['arg2']), file=f)
      print(r['doc_id'] + '\tlabel\t' + r['label'], file=f)
      print(file=f)


def read_word_embed(vocab):
  """Google word2vec 300 dim"""
  existing_embed =  utils.load_word2vec(WORD2VEC_FILE, vocab)
  word_embed = [None]*len(vocab)
  for word in vocab:
    index = vocab[word]
    try:
      embed = np.array(existing_embed[word], dtype=np.float32)
    except KeyError:
      embed = np.random.uniform(low=-0.25, high=0.25, size=[300]).tolist()
    word_embed[index] = embed
  return np.array(word_embed)



def read_vocab(relations, n=None):
  vocab = {'</s>': 0, '<UNK>':1, '<s>':2}
  cnt = Counter()
  for relation in relations:
    for token in relation['token']:
      cnt[token] += 1

  if n is None:
    cnt_list = cnt.most_common()
  else:
    cnt_list = cnt.most_common(n-3)
  print(cnt_list[-10:])
  index = 3
  for v, c in cnt_list:
    vocab[v] = index
    index += 1 
  return vocab

def read_entity_set(relations, bio=False):
  entity_set = {}
  cnt = Counter()
  for relation in relations:
    for entity_type in relation['entity_type']:
      if bio or entity_type == 'O':
        cnt[entity_type] += 1
      else:
        cnt[entity_type[2:]] += 1 # remove B- I-
  cnt_list = cnt.most_common()
  print(cnt_list)
  index = 0
  for e, c in cnt_list:
    entity_set[e] = index
    index += 1 
  return entity_set

def read_label_set(relations, include_subtype):
  labels = {}
  cnt = Counter()
  for relation in relations:
    if include_subtype or relation['label'] == 'OTHER':
      cnt[relation['label']] += 1
    else:
      label = relation['label']
      if label.endswith('(e1,e2)') or label.endswith('(e2,e1)'):
        label_prefix = label[:-7]
        label_suffix = label[-7:]
        rel_type, unused_subtype = label_prefix.split('.')
        rel_type = rel_type + label_suffix
      else:
        rel_type, unused_subtype = label.split('.')
      cnt[rel_type] += 1
  cnt_list = cnt.most_common()
  print(cnt_list)
  index = 0
  if not include_subtype:
    # use predefined index
    for l, c in cnt_list:
      if l in PREDEFINED_LABEL_SET:
        labels[l] = PREDEFINED_LABEL_SET[l]
      else:
        print(l, 'not in predefined label set')
  else:
    for l, c in cnt_list:
     labels[l] = index
     index += 1
  return labels

def read_doc_ids(relations):
  doc_ids = []
  cnt = Counter()
  for relation in relations:
    cnt[relation['doc_id']] += 1
  cnt_list = cnt.most_common()
  print(cnt_list[:10])
  for d,v in cnt_list:
    doc_ids += [d]
  return doc_ids

def read_pos_doc_ids(relations):
  pos_relations = []
  for relation in relations:
    if relation['label'] != 'Other':
      pos_relations += [relation]
  return read_doc_ids(pos_relations)

def textToId(tokens, vocab):
  ids = []
  for token in tokens:
    try:
      ids += [vocab[token]]
    except KeyError:
      ids += [1]  # UNK
  return ids

def convert_type_args_only(entity_types, entity_set, arg1, arg2, bio=False):
  ids = []
  index = 0
  if entity_types[arg1] == 'O' or entity_types[arg2] == 'O':
    raise ValueError('args are not entities')
  for entity_type in entity_types:
    if index == arg1 or index == arg2:
      if bio: 
        ids += [entity_set[entity_type]]
      else:
        ids += [entity_set[entity_type[2:]]] # remove B- I-
    else:
      ids += [entity_set['O']]
    index += 1
  return ids

def convert_type_args_extent(entity_types, entity_set, arg1, arg2, bio=False):
  ids = convert_type_args_only(entity_types, entity_set, arg1, arg2, bio)
  for i in range(len(ids)):
    if ids[i] != entity_set['O'] and entity_types[i].startswith('I-'):
      j = i -1
      while entity_types[j].startswith('I-'):
        j -= 1
      for k in range(j, i):
        ids[k] = ids[i]
#      print(entity_types)  
#      print(ids)
  return ids

def convert_type(entity_types, entity_set, arg1, arg2, bio=False):
  ids = []
  if entity_types[arg1] == 'O' or entity_types[arg2] == 'O':
    raise ValueError('args are not entities')
  for entity_type in entity_types: 
    if bio or entity_type == 'O':
      ids += [entity_set[entity_type]]
    else:
      ids += [entity_set[entity_type[2:]]] # remove B- I-
  return ids

def convert_pos(arg_index, args, max_len):
  """Only mark of the first token of the argument"""
  pos_indices = []
  for i in xrange(len(args)):
    pos = i-arg_index + max_len
    if pos < 0: pos =0
    if pos >=max_len*2: pos = max_len*2-1
    pos_indices +=[pos]
  return pos_indices

def convert_label(label, include_subtype):
  """e.g. return phyical or physical.locatednear
     return employment(e1, e2) or employment.subsidiary(e1, 2)
  """
  if include_subtype: return label

  if label == 'OTHER':
    return label
  if label.endswith('(e1,e2)') or label.endswith('(e2,e1)'):
    label_prefix = label[:-7]
    label_suffix = label[-7:]
    rel_type, unused_subtype = label_prefix.split('.')
    rel_type = rel_type + label_suffix
  else:
    rel_type, unused_subtype = label.split('.')
  return rel_type

def convert_to_directional(relations):
  d_relations = []
  is_pos = {}
  for relation in relations:
    arg1 = len(relation['arg1']) - relation['arg1'][::-1].index('1') - 1
    arg2 = len(relation['arg2']) - relation['arg2'][::-1].index('2') - 1
    if arg1 > arg2 and relation['label'] == 'OTHER':
      continue # this should have been covered with arguments swap
    if arg1 > arg2 and relation['label'] != 'OTHER':
      arg1_list = ['1' if x == '2' else x for x in relation['arg2']]
      arg2_list = ['2' if x == '1' else x for x in relation['arg1']]
      relation['arg1'] = arg1_list
      relation['arg2'] = arg2_list
      if relation['label'].startswith('physical') or relation['label'].startswith('personalsocial'):
        pass # do nothing
      else:
        relation['label'] = relation['label'] + '(e2,e1)'
    elif arg1 <= arg2 and relation['label'] != 'OTHER':
      if relation['label'].startswith('physical') or relation['label'].startswith('personalsocial'):
        pass # do nothing
      else:
        relation['label'] = relation['label'] + '(e1,e2)'
    relation['id'] = hash(' '.join(relation['token'] + relation['arg1'] + relation['arg2']))
    if relation['label'] != 'OTHER':
      is_pos[relation['id']] = True
      
    # do nothing if arg1 < arg2 and label is OTHER
    d_relations += [relation]
  #remove conflict OTHER examples
  d_clean_relations = []
  for r in d_relations:
    if r['id'] in is_pos and r['label'] == 'OTHER':
      continue
    d_clean_relations += [r]

  return d_clean_relations

def process_relation(relations, vocab, entity_set, label_set, max_len, include_subtype=False):
  feature_lists = []
  data_error_num = 0
  for relation in relations:
    arg1 = len(relation['arg1']) - relation['arg1'][::-1].index('1') - 1
    arg2 = len(relation['arg2']) - relation['arg2'][::-1].index('2') - 1
    feature = {}
    feature['word'] = textToId(relation['token'], vocab)
    try:
      feature['type'] = convert_type_args_only(relation['entity_type'], entity_set, arg1=arg1, arg2=arg2)
    except ValueError:
      if relation['label'] != 'OTHER':
        print(relation['label'])
        print(relation['token'])
        print(relation['arg1'])
        print(relation['arg2'])
        print(relation['entity_type'])
      else:
        data_error_num += 1
        continue
    feature['pos1'] = convert_pos(arg1, relation['arg1'], max_len)
    feature['pos2'] = convert_pos(arg2, relation['arg2'], max_len)
    feature['y'] = label_set[convert_label(relation['label'], include_subtype)]
    feature_lists += [feature]
  print('example with inconsistent arg and entity type', data_error_num) 
  return feature_lists

def split_doc_ids(doc_ids):
  random.Random(42).shuffle(doc_ids)
  dev_len = int(len(doc_ids)*0.1)
  # train, dev, test
  train = set(doc_ids[dev_len*2:])
  dev = set(doc_ids[:dev_len])
  test = set(doc_ids[dev_len:dev_len*2])
  random.seed()
  return train, dev, test

def split_doc_ids_portion(doc_ids, split=[0.8, 0.1, 0.1]):
  """split [train, dev, test]"""
  random.Random(42).shuffle(doc_ids)
  t = len(doc_ids)
  # train, dev, test
  train = set(doc_ids[int(t*(1- split[0])):])
  dev = set(doc_ids[:int(t*split[1])])
  test = set(doc_ids[int(t*split[1]):int(t*(split[1]+split[2]))])
  random.seed()
  return train, dev, test

def split_dataset_evenly_by_types(relations, split=[0.8, 0.1, 0.1]):
  r_by_type = {}
  for r in relations:
    if r['label'] not in r_by_type:
      r_by_type[r['label']] = []
    r_by_type[r['label']] += [r]
  train, dev, test = [], [] ,[]
  for r_type, r_list in r_by_type.iteritems():
    t = len(r_list)
    train += r_list[: int(t*split[0])]
    dev += r_list[int(t*split[0]):int(t*(split[0]+split[1]))]
    test += r_list[int(t*(split[0]+split[1])):int(t*(split[0]+split[1]+split[2]))]
  random.seed(42)
  random.shuffle(train)
  random.shuffle(dev)
  random.shuffle(test)
  random.seed()
  return train, dev, test

def split_dataset(relations, doc_ids, split=[0.8, 0.1, 0.1]):
  train_ids, dev_ids, test_ids = split_doc_ids_portion(doc_ids,
                                split=split)
  train, dev, test = [], [], []
  for relation in relations:
    if relation['doc_id'] in train_ids:
      train += [relation]
      continue
    if relation['doc_id'] in dev_ids:
      dev += [relation]
      continue
    if relation['doc_id'] in test_ids:
      test += [relation]

  return train, dev, test

def write_doc_split(train, dev, test):
  def write_doc_ids(filename, relations):
    doc_ids = set()
    for r in relations:
      doc_ids.add(r['doc_id'])
    with open(filename, 'w') as output:
      for doc_id in doc_ids:
        print(doc_id, file=output)
  write_doc_ids(DOC_ID_PATH + 'train_doc_ids', train)
  write_doc_ids(DOC_ID_PATH + 'dev_doc_ids', dev)
  write_doc_ids(DOC_ID_PATH + 'test_doc_ids', test)


def produce_doc_ids():
  relations = read_relation_candidates(DATA_PATH)
  doc_ids = read_doc_ids(relations)
  train, dev, test = split_dataset(relations, doc_ids)
  write_doc_split(train, dev, test)

def process_main_subtype_names(label_vocab, vocab, word_embed):
  names, label_name_vocab, label_name_embed = process_label_names(label_vocab, vocab, word_embed, True)
  split_id = label_name_vocab['/']
  main = []
  subtype = []
  for name in names:
    index = name[1:].index(split_id)
    main += [name[1:index+1]]
    subtype += [name[index+2:]]
  return main, subtype, label_name_vocab, label_name_embed

def process_label_names(label_vocab, vocab, word_embed, subtype=False, binary=False, extended_tag=False, append_tag=False):
  if binary:
    names_split = {
        'OTHER': ['/', 'other'],
        'relation': ['/', 'relation'],
    }
    label_name_vocab = {'</s>': 0,'<s>':1, 'other':2, 'relation':3}
    final_list = [[label_name_vocab['other']],[label_name_vocab['relation']]]
    label_name_embed = [word_embed[vocab['<s>']], word_embed[vocab['</s>']],
                        word_embed[vocab['other']], word_embed[vocab['relation']]]
    return final_list, label_name_vocab, np.array(label_name_embed, dtype=np.float32)

  final_list = [0]*len(label_vocab)
  def random_vector(size):
    return np.random.uniform(low=-0.25, high=0.25, size=[size])
  label_name_vocab = {'</s>':0, '<s>':1, '<ERE>': 2}
  label_name_embed = [word_embed[vocab['<s>']], word_embed[vocab['</s>']], random_vector(len(word_embed[0]))]
  for name, index in label_vocab.iteritems():
    name_desc = []
    if subtype:
      names_split = {
          'generalaffiliation.more': ['/', 'general', 'affiliation', '/', 'member', 'origin', 'religion', 'ethnicity'], #ethnicity out of vocab
          'generalaffiliation.opra': ['/', 'general', 'affiliation', '/', 'organization', 'political', 'religious', 'affiliation'],
          'orgaffiliation.employmentmembership': ['/', 'organization', 'affiliation', '/', 'employment', 'membership'],
          'orgaffiliation.founder': ['/', 'organization', 'affiliation', '/', 'founder'],
          'orgaffiliation.investorshareholder': ['/', 'organization', 'affiliation', '/', 'investor', 'shareholder'],
          'orgaffiliation.leadership': ['/', 'organization', 'affiliation', '/', 'leadership'],
          'orgaffiliation.ownership': ['/', 'organization', 'affiliation', '/', 'ownership'],
          'orgaffiliation.studentalum': ['/', 'organization', 'affiliation', '/', 'student', 'alumni'],
          'OTHER': ['/', 'other'],
          'partwhole.membership': ['/', 'part', 'whole', '/', 'membership'],
          'partwhole.subsidiary': ['/', 'part', 'whole', '/', 'subsidiary'],
          'personalsocial.business': ['/', 'personal', 'social', '/', 'business'],
          'personalsocial.family': ['/', 'personal', 'social', '/', 'family'],
          'personalsocial.role': ['/', 'personal', 'social', '/', 'role'],
          'personalsocial.unspecified': ['/', 'personal', 'social', '/', 'unspecified'],
          'physical.locatednear': ['/', 'physical', '/', 'located', 'near'],
          'physical.orgheadquarter': ['/', 'physical', '/', 'organization', 'headquarter'],
          'physical.orglocationorigin': ['/', 'physical', '/', 'organization', 'location', 'origin'],
          'physical.resident': ['/', 'physical', '/', 'resident'],
      }
    else:
      if extended_tag:
        names_split = {
          'OTHER': ['/', 'other'],
          'partwhole': ['/', 'part', 'whole', '<ERE_partwhole>'],
          'orgaffiliation': ['/', 'organization', 'affiliation', '<ERE_orgaffiliation>'],
          'generalaffiliation': ['/', 'general', 'affiliation', '<ERE_generalaffiliation>'],
          'personalsocial': ['/', 'personal', 'social', '<ERE_personalsocial>'],
          'physical': ['/', 'physical', '<ERE_physical>'],
        }
      else:
        names_split = {
          'OTHER': ['/', 'other'],
          'partwhole(e1,e2)': ['/', 'part', 'whole', '<right>'],
          'partwhole(e2,e1)': ['/', 'part', 'whole', '<left>'],
          'orgaffiliation(e1,e2)': ['/', 'organization', 'affiliation', '<right>'],
          'orgaffiliation(e2,e1)': ['/', 'organization', 'affiliation', '<left>'], 
          'generalaffiliation(e1,e2)': ['/', 'general', 'affiliation', '<right>'],
          'generalaffiliation(e2,e1)': ['/', 'general', 'affiliation', '<left>'],
          'personalsocial': ['/', 'personal', 'social'],
          'physical': ['/', 'physical'],
        }

    for key in names_split:
      if append_tag:
        names_split[key] += ['<ERE>']
    for token in names_split[name]:
      if token == '/':
        continue
        #token = '<s>'
      if token not in label_name_vocab:
        label_name_vocab[token] = len(label_name_vocab)
        if token not in vocab:
          label_name_embed += [random_vector(len(word_embed[0]))]
        else:
          label_name_embed += [word_embed[vocab[token]]]
      name_desc += [label_name_vocab[token]]
    final_list[index] = name_desc
  return final_list, label_name_vocab, np.array(label_name_embed, dtype=np.float32)

def convert_to_directional_dataset(output_file,relation_file=DATA_PATH):
  relations = read_relation_candidates(relation_file)
  d_relations = convert_to_directional(relations)
  write_relation_candidates(d_relations, output_file)


def load_dataset(relation_file=DATA_PATH, vocab_size=25227, max_len=50, include_subtype=False,
                 split=[0.8, 0.1, 0.1]):
  relations = read_relation_candidates(relation_file)
  
  entity_set = read_entity_set(relations)
  print('Count all relations')
  label_set = read_label_set(relations, include_subtype)

  doc_ids = read_doc_ids(relations)
  train, dev, test = split_dataset(relations, doc_ids, split=split)
  # train, dev, test = split_dataset_evenly_by_types(relations, split=[0.4, 0.2, 0.4])

  vocab = read_vocab(train, n=vocab_size)
  print('Count all relatoins in train')
  read_label_set(train, include_subtype)
  print('Count all relations in test')
  read_label_set(test, include_subtype)
  print('label_set', label_set)
  train = process_relation(train, vocab, entity_set, label_set, max_len, include_subtype)
  dev = process_relation(dev, vocab, entity_set, label_set, max_len, include_subtype)
  test = process_relation(test, vocab, entity_set, label_set, max_len, include_subtype)
  print('entity set', entity_set)
  word_embed = read_word_embed(vocab)
  # label_names, names_vocab, names_embed = process_label_names(label_set, vocab, word_embed, include_subtype)
  # print('label names', label_names, names_vocab)
  return train, dev, test, vocab, label_set, word_embed

def pick_relation_type(train, dev, test, picked_types=[]):
  #TODO
  train[1] = np.argmax(train[1], axis=1)
  for i in xrange(len(train[1])):
    if train[1][i] not in picked_types:
      train[1][i] = 0 # other
  train[1] = to_categorical(train[1], nb_classes=len(picked_types)+1)


def load_entity_set(relation_file=DATA_PATH):
  relations = read_relation_candidates(relation_file)
  return read_entity_set(relations)

def count_sentence_length(relations):
  cnt = Counter()
  avg_sent_len = 0
  for relation in relations:
    length = len(relation['word'])
    avg_sent_len += length
    if length < 10:
      cnt['sent_len <10'] += 1
    elif length < 20:
      cnt['10 <= sent_len < 20'] += 1
    elif length < 50:
      cnt['20 <= sent_len < 50'] += 1
    elif length < 100:
      cnt['50 <= sent_len < 100'] +=1
    else:
      cnt['sent_len >= 100'] += 1
  avg_sent_len /= float(len(relations))
  return cnt, avg_sent_len


if __name__ == '__main__':
  # convert_to_directional_dataset(output_file='./data/ERE/ere_directional.txt')
  produce_doc_ids()
  print('doc_ids created')
  train, dev, test, vocab, label_set, word_embed = load_dataset(include_subtype=False)
  print(len(train), len(dev), len(test), len(vocab), len(label_set))
  print(word_embed.shape)

  cnt, avg_sent_len = count_sentence_length(train+test+dev)
  print(avg_sent_len, cnt.most_common())  
  cnt, avg_sent_len = count_sentence_length(train)
  print(avg_sent_len, cnt.most_common())
  cnt, avg_sent_len = count_sentence_length(dev)
  print(avg_sent_len, cnt.most_common())
  cnt, avg_sent_len = count_sentence_length(test)
  print(avg_sent_len, cnt.most_common())
