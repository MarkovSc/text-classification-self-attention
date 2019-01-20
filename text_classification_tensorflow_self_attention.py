
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import time
import math
import datetime
import os

import pickle
dic={}
with open('./vocab_pickle_file', 'rb') as f:
  dic = pickle.load(f)

#in inteactive jupyter env, this downloads the glove embeddings. You can also remove the '!' and run it through cmd prompt to download  
!wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip 

import numpy as np
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

words_to_index, index_to_words, word_to_vec_map = read_glove_vecs('./glove.6B.300d.txt')

import keras

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #common = set(string.split()).intersection(person_list)
    #if len(common) > 0:
    #    for i in list(common):
    #        if i in ['ms', 'mrs', 'mr', 'dr']:
    #            string = re.sub(r'{} ?[A-Za-z]+ '.format(i), 'person', string)
    #        else:
    #            string = string.replace(i, ' person ')
    #print (string, '\n')
    string = string.strip().lower()
    #common = set(sent.split()).intersection(lst)
    #if len(common) > 0:
    #    for i in list(common):
    #        if i in ['ms', 'mrs', 'mr', 'dr']:
    #            string = re.sub(r'{} ?[A-Za-z]+ '.format(i), 'person ', string)
    #        else:
    #            string = string.replace(i, ' person ')
    #print (string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r',', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    if 'let\'s' in string:
      string = re.sub(r'let\'s', 'let us', string)
    if 'lets' in string:
      string = re.sub(r'lets', 'let us', string)
      
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    if 'wont ' in string:
      string = re.sub(r"won\'?t", "will not", string)
    if 'won\'t ' in string:
      string = re.sub(r"won\'?t", "will not", string)
    if 'dont ' in string:
      string = re.sub(r"don\'?t", "do not", string)
    if 'don\'t ' in string:
      string = re.sub(r"don\'?t", "do not", string)
    
    if 'cant ' in string:
      string = re.sub(r"can\'?t", " can not", string)
    if 'can\'t ' in string:
      string = re.sub(r"can\'?t", " can not", string)

    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    #string = re.sub(r"\'", '', string)

    return string.strip()

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = pd.read_csv(positive_data_file, sep = '\n', header = None)[0]
    positive_examples = [s.strip() for s in positive_examples]

    negative_examples = pd.read_csv(negative_data_file, sep = '\n', header = None)[0]
    negative_examples = [s.strip() for s in negative_examples]
    print (len(positive_examples), len(negative_examples))
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [ 0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


x_text, y = load_data_and_labels('positives.txt', 'negatives.txt')

df = pd.DataFrame()
df['text'] = x_text
df['target'] = y

np.random.seed(10)
df = df.sample(frac = 1)
ls = list(df.text)
y = df.target

df = (df[df['text'] != ''])

y = list(y)

words = list(set(' '.join(ls).split()))
count = 20654
for i in words:
  if i not in dic.keys():
    dic[i] = count
    count+=1

ct1 = 0
maxlen = 200
arr = np.zeros((len(ls), maxlen))
for pos,sent in enumerate(ls):
  sent1 = sent.split()
  ct2 = 0
  for word in (sent1):
    arr[ct1, ct2] = dic[word]
    ct2+=1
  ct1+=1

vocab_size = len(dic) + 1


dev_sample_index = -1 * int(0.2 * float(len(arr)))
x_train, x_dev = arr[:dev_sample_index], arr[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

embed_matrix1 = np.zeros((vocab_size, 300))
for word,index in dic.items():
  try:
    embed_matrix1[index, :] = word_to_vec_map[word]
  except:
    embed_matrix1[index, :] = np.random.uniform(-1, 1, 300)

num_epochs = 40
minibatch_size = 64

m = x_train.shape[0]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def random_mini_batches(x_tra, y_tra, mini_batch_size = 64):
    mini_batches = []

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = x_tra[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = y_tra[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        print ()
        mini_batch_X = x_tra[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = y_tra[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
  

with tf.Graph().as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  
  with sess.as_default():
    tf.set_random_seed(0)
    with tf.name_scope('arr_placeholder'):
      arr_placeholder = tf.placeholder(dtype = tf.int64, shape = [None, maxlen], name = 'inp_array')
      labels = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = 'labels')
    with tf.name_scope('dropout_keep_prob'):
      dropout_keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    
    with tf.name_scope('embed'):
      with tf.device('/gpu:0'):
        W = tf.get_variable(name = 'embed_matrix', shape = embed_matrix1.shape, initializer = tf.constant_initializer(embed_matrix1), trainable = False)
        embeddings_out = tf.nn.embedding_lookup(W, arr_placeholder, name = 'lookup')


    with tf.name_scope('GRU_2'):
      with tf.device('/gpu:0'):
        gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences = True, name = 'GRU_2'))(embeddings_out)

    with tf.name_scope('annotation_vector'):
      with tf.device('/gpu:0'):
        W_s1 = tf.get_variable(shape = (150, 128), name = 'W_s1', initializer = tf.contrib.layers.xavier_initializer())
        W_s2 = tf.get_variable(shape = (6, 150), name = 'W_s2', initializer = tf.contrib.layers.xavier_initializer())
        with tf.device('/cpu:0'):
          A = tf.nn.softmax(tf.transpose(tf.tensordot(W_s2, tf.nn.tanh(tf.transpose(tf.tensordot(W_s1, tf.transpose(gru2, [0,2,1]), axes = [1, 1]), [1,0,2]), name = 'A'), axes = [1,1]), [1,0,2]))
        M = tf.matmul(A, gru2, name = 'M')
        
    with tf.name_scope('linear_1'):
      intermediate = tf.reduce_max(M, axis = 1, name = 'intermediate')
      linear_1 = tf.layers.dense(intermediate, 64, activation = tf.nn.relu, name = 'linear1')

    with tf.name_scope('dropout_2'):
      dropout_2 = tf.nn.dropout(linear_1, keep_prob = dropout_keep_prob, name = 'dropout2')

    with tf.name_scope('linear_2'):
      linear_2 = tf.layers.dense(dropout_2, 32, activation = tf.nn.relu, name = 'linear2')
    with tf.name_scope('logits'):
      logits = tf.layers.dense(linear_2, 1, name = 'preds')
      #scores = tf.argmax(logits, 1, name = 'scores')
    with tf.name_scope('loss'):
      difference = tf.subtract(tf.matmul(A, A, transpose_b = True), tf.eye(6), name = 'difference')
      penalty = tf.norm(difference, axis = [-2,-1], name = 'penalty')
      print (penalty.get_shape())
      loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = logits, targets = labels, pos_weight = 2.1, name = 'loss_fn') ) + tf.reduce_mean(penalty)


    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0025).minimize(loss)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 40)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    current_step = 0
    for epoch in range(num_epochs):

      current_step+=1
      predictionstrain = []
      #predictions_dev = []
      epoch_cost = 0.                       # Defines a cost related to an epoch
      num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
      minibatches = random_mini_batches(x_train, y_train, minibatch_size)
      mini_count = 0
      for minibatch in minibatches:
          # Select a minibatch
          (minibatch_x, minibatch_y) = minibatch
          #print (len(minibatch_x))
          mini_count +=1
          feed_dict = {
          arr_placeholder: minibatch_x ,
          labels: np.array(minibatch_y).reshape(len(minibatch_y), 1),
          dropout_keep_prob: 0.75}
          _, loss1 = sess.run(
            #[train_op, global_step, train_summary_op, 
              [optimizer, loss],
            feed_dict)
          
          epoch_cost += loss1/num_minibatches
          if mini_count%50 == 0:
            print ('{} minibatches over'.format(mini_count))

      path = saver.save(sess, checkpoint_prefix, global_step=current_step)
   
      predictions_dev = []
      for i in  sess.run(tf.nn.sigmoid(logits), feed_dict = {arr_placeholder : x_dev, labels : np.array(y_dev).reshape(len(y_dev), 1), dropout_keep_prob : 1.0}):
        if i > 0.5:
          predictions_dev.append(1.0)
        else:
          predictions_dev.append(0.0)
      if sum(predictions_dev) == 0 : print ('\nall zeros in dev set prediction')
      c = tf.cast(tf.convert_to_tensor(predictions_dev), 'float')
      d = (tf.cast(tf.convert_to_tensor([i for i in y_dev]), 'float'))

      correct_predictions_dev = sess.run(tf.equal(c, d))
      accuracy_dev = sess.run(tf.reduce_mean(tf.cast(correct_predictions_dev, "float"), name="accuracy_dev"))
      if epoch % 1 == 0:
          print ("Cost after epoch %i: %f" % (epoch+1, epoch_cost))
          #print ("Accuracy after epoch %i : %f" % (epoch+1, (accuracy_train)))#, feed_dict = {arr_placeholder : x_train,  str_placeholder : ls[:4731], dropout_keep_prob : 1.0})))
          print ("Dev Accuracy after epoch %i : %f" % (epoch+1, (accuracy_dev)))# feed_dict = {arr_placeholder : x_dev, str_placeholder : ls[4731:], dropout_keep_prob : 1.0})))          
          all_preds = predictions_dev
          labels_ =  y_dev
          #if (precision_score(labels, all_preds) > 0.87 and f1_score(labels, all_preds) > 0.85) or (recall_score(labels, all_preds) > 0.87 and f1_score(labels, all_preds) > 0.85) :
          print ('\nModel: {}'.format(epoch+1))
          print ('precision: ', precision_score(labels_, all_preds))
          print ('recall: ', recall_score(labels_, all_preds))
          print ('f1 :', f1_score(labels_, all_preds))
          print ('accuracy :', accuracy_score(labels_, all_preds))
          print (confusion_matrix(labels_, all_preds))
          print ('\n')
      print("\nSaved model checkpoint to {}\n".format(path))

