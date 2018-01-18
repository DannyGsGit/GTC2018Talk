
# coding: utf-8

# In[1]:


import os, math, csv, re, itertools, collections
import numpy as np
import pandas as pd
from collections import Counter
import jellyfish as jyfs
import datetime, time
import pickle
import matplotlib.pyplot as plt
import inflect
from random import *
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

print("Starting script")

# # Tokenize and Train Notebook
# This notebook will take a text corpus and use it for training.

# In[2]:


# Import the dataset
with open('./tempfiles/cleanParts.pkl', 'rb') as f:
   words = pickle.load(f)


# In[3]:


# Inspect the dataset
print(type(words))
words[0:10]


# # Generate Training Batches
# This function will be called during training to generate minibatches for the skip-gram model.

# In[4]:


# Map words to indices
word2index_map = {}
index = 0
for sent in words:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)

print("Vocab size:", vocabulary_size)
# print("Word Index:", index2word_map)


# In[5]:


# Inspect the top of the dictionary
dict(list(index2word_map.items())[0:5])


# In[6]:


# Define a function to generate skip-grams
# Initialize the skip-gram pairs list
skip_gram_pairs = []

# Set the skip-gram window size
window_size = 2

for sent in words:
    tokenized_sent = sent.split()
    # Set the target index
    for tgt_idx in range(0, len(tokenized_sent)):
        # Set range for the sentence
        max_idx = len(tokenized_sent) - 1

        # Define range around target
        lo_idx = max(tgt_idx - window_size, 0)
        hi_idx = min(tgt_idx + window_size, max_idx) + 1

        # List the indices in the skip-gram outputs (removing target index)
        number_list = range(lo_idx, hi_idx)
        output_matches = list(filter(lambda x: x != tgt_idx, number_list))

        # Generate skip-gram pairs
        pairs = [[word2index_map[tokenized_sent[tgt_idx]], word2index_map[tokenized_sent[out]]] for out in output_matches]
        # print(pairs)

        for p in pairs:
            skip_gram_pairs.append(p)


# In[7]:


# Inspect some output:
skip_gram_pairs[0:12]


# Now define a function to sample batches from the skipgram pairs during training.

# In[21]:


def get_skipgram_batch(start_index, end_index):
    instance_indices = list(range(len(skip_gram_pairs)))
    # np.random.shuffle(instance_indices)
    batch = instance_indices[start_index:end_index]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y


# In[22]:


# batch example
x_batch, y_batch = get_skipgram_batch(0,8)

print("X Batch: ", [index2word_map[word] for word in x_batch])
print("Y Batch: ", [index2word_map[word[0]] for word in y_batch])


# # Training

# ## ========= To Do Training ==================
# Add the following features:
# * Number of epochs to train
# ## =========================================

# In[39]:


batch_size = 256
embedding_dimension = 128
negative_samples = 128
n_iterations = int(round(2 * len(skip_gram_pairs) / batch_size,0))
LOG_DIR = "logs/word2vec_cab"


# In[34]:


print("There are ", len(skip_gram_pairs), " skip-gram pairs")
print("The chosen iteration and batch size parameters will yield ",
     round((batch_size * n_iterations)/len(skip_gram_pairs),2), " epochs.")


# In[35]:


graph = tf.Graph()

# This may work with GPU
with graph.as_default(), tf.device('/cpu:0'), tf.name_scope("embeddings"):
    # Input data, labels
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    
    # Embedding lookup table
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension],
                          -1.0, 1.0), name='embedding')
    # This is essentialy a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
    # Create variables for the NCE loss
    nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_dimension],
                                stddev=1.0 / math.sqrt(embedding_dimension)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    with tf.device('/cpu:0'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           inputs=embed,
                           labels=train_labels,
                           num_sampled=negative_samples,
                           num_classes=vocabulary_size))
        tf.summary.scalar("NCE_loss", loss)
    
    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                              global_step=global_step,
                                              decay_steps=1000,
                                              decay_rate=0.95,
                                              staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
    merged = tf.summary.merge_all()


# In[ ]:


max_index = len(skip_gram_pairs)
start_index = 0
end_index = min(start_index + batch_size, max_index)

with tf.Session(graph=graph) as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR,
                                         graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)

    tf.global_variables_initializer().run()

    for step in range(n_iterations):
        x_batch, y_batch = get_skipgram_batch(start_index, end_index)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={train_inputs: x_batch,
                                         train_labels: y_batch})
        train_writer.add_summary(summary, step)
        
        
        if start_index >= max_index:
            start_index = 0
        else:
            start_index = start_index + batch_size + 1
        
        end_index = min(start_index + batch_size, max_index)
        

        if step % 100 == 0:
            print("Completed ", step, " of ", n_iterations)
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss,
                                  feed_dict={train_inputs: x_batch,
                                             train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))

    # Normalize embeddings before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

