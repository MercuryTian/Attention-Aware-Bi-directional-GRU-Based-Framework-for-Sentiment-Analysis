# encoding: utf-8
# @author: Zhengxi Tian
# email: zhengxi.tian@hotmail.com

import os
import codecs
import re
import tensorflow
import keras
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding, Reshape
from keras.layers import LSTM, GRU
from keras import optimizers
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model

# Initialization
'''
X.shape = (m, seq_length)
Y.shape = (m, label)
label: [pos, neg], possibility of polar sentiment
m = 4530
'''
X = []
Y = None
X_train = None
X_test = None
Y_train = None
Y_test = None
X_val = None
Y_val = None
MAX_LENGTH = 551

# Pre-processing texts
filepath = '/home/wenger/zhengxitian/atGRU/CustomerReviews/'

def __travDir__(filepath):
    filelist = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        filelist.append(child)
    for n in filelist:
        __preProduceFile__(n)

def __preProduceFile__(filename):
    global X, Y
    fopen = codecs.open(filename, 'r+', 'utf-8', errors='ignore')
    for eachLine in fopen.readlines():
        label = 0
        counts = 0
        if '##' in eachLine:
            if eachLine.startswith('##'):
                continue
            else:
                tmp = eachLine.split('##')
                t = re.split('\[|\]|\,', tmp[0])
                for i in t:
                    if i=='-1' or i== '+1' or i=='+2' or i=='-2' or i=='+3' or i=='-3':
                        label += int(i)
                        counts += 1
                if counts != 0:
                    score = label / counts
                    # print('this score is:', score)
                else:
                    print('the wrong:',i)
                    score = 0
            
                if Y is None:
                    if score > 0:
                        Y = np.array([1,0])
                    elif score < 0:
                        Y = np.array([0,1])
                else:
                    if score > 0:
                        Y = np.row_stack((Y, np.array([1,0])))
                    elif score < 0:
                        Y = np.row_stack((Y, np.array([0,1])))
                if score != 0:
                    X.append(tmp[1])
    fopen.close()

def __findMax__():
    MAX_LENGTH = 0
    for i in X:
        tmp = i.strip(' ').split()
        counts = len(i)
        if counts >= MAX_LENGTH:
            MAX_LENGTH = counts
    print('The max length of a review is:', MAX_LENGTH)
    return MAX_LENGTH

def __randomSplit__(X, Y):
    X = np.array(X)
    shuffle_index = np.random.permutation(np.arange(len(Y)))
    X_new = X[shuffle_index]
    Y_new = Y[shuffle_index]
    X_train = X_new[0:3623]
    Y_train = Y_new[0:3623]
    X_val = X_new[3624:4076]
    Y_val = Y_new[3624:4076]
    X_test = X_new[4077:4529]
    Y_test = Y_new[4077:4529]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Start producing
__travDir__(filepath)
X_train, Y_train, X_val, Y_val, X_test, Y_test = __randomSplit__(X, Y)
# MAX_LENGTH = __findMax__

# Testing initialization
print('X.shape:', len(X))
print('Y.shape:', len(Y))
for i in range(5):
    print('X tarin:', X_train[i])
    print('Y tarin:', Y_train[i])
print('\n')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
for i in range(5):
    print('X test:', X_test[i])
    print('Y test:', Y_test[i])


# Parameters
'''
When changing the embedding file:
remember to change the EMBEDDING_DIM and filename in fopen
'''
EMBEDDING_DIM = 200
MAX_NUM_WORDS = 1200000

'''
GloVe Embeddings - Twitter - 2B tweets, 27B tokens, 1.2M vocab
embeddig_matrix: pre-trained word vectors
word_index: dictionary of word vectors
Concat: concatenate training and testing sets, in order to build the dictionary of embedding vectors
sequences: parsing results of training and testing sets
data: training sets exchange to embeding vectors matrix
'''
GLOVE_DIR = '/home/wenger/zhengxitian/atGRU/'
embedding_index = {}
fopen = codecs.open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt'), 'r', 'utf-8', errors='ignore')

for eachLine in fopen.readlines():
    # First element in each line is the word
    values = eachLine.split()
    word = values[0]
    # Word vectors
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
fopen.close()
print('Found %s word vectors.' % len(embedding_index))

# Vectoize the TRAINING text samples into a 2D integer tensor
X_train = X_train.tolist()
X_test = X_test.tolist()
X_val = X_val.tolist()
Concat = X_train + X_test + X_val
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer.fit_on_texts(Concat)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Auto filled with 0
data_train = pad_sequences(sequences_train, maxlen = MAX_LENGTH)
data_test = pad_sequences(sequences_test, maxlen = MAX_LENGTH)
data_val = pad_sequences(sequences_val, maxlen = MAX_LENGTH)

# Prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)+1) 
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# Load pre-trained word embeddings into an Embedding Layer
embedding_layer = Embedding(num_words, 
                            EMBEDDING_DIM, 
                            weights=[embedding_matrix], 
                            input_length=MAX_LENGTH,
                            trainable=False)
print('Trainning embedding model.')


# Attention
'''
If SINGLE_ATTENTION_VECTOR = true, 
the attention vector is shared across the input_dimensions where the attention is applied.
'''
from keras.layers import multiply

SINGLE_ATTENTION_VECTOR = False
TIME_STEPS = MAX_LENGTH

def __attention3DBlock__(inputs):
    '''
    Input shape = (batch_size, time_steps, input_dim)
    '''
    input_dim = int(inputs.shape[2])
    a = Permute((2,1))(inputs)  # (batch_size, input_dim, time_steps)
    a = Reshape((input_dim, TIME_STEPS))(a) # (batch_size, input_dim, time_steps)
    a = Dense(TIME_STEPS, activation='softmax')(a) # (batch_size, input_dim, time_steps)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a) # (batch_size, input_dim, time_steps)
    a_probs = Permute((2,1), name='attention_vector')(a) # (batch_size, time_steps, input_dim)
    output_attention_mul = multiply([inputs, a_probs], name='attenton_mul') # (batch_size, time_steps, input_dim)

    return output_attention_mul


# Model
from keras import regularizers

# Parameters
'''
n_x: hidden state size of Bi-GRU
regularization: L2 normalization
optimization: AdaGrad/Adam
time_steps = MAX_LENGTH = 551
'''
batch_size = 25
momentum = 0.9
l2_regularization = 0.001
learning_rate = 0.01
n_x = 32   
epochs = 20
time_steps = MAX_LENGTH

# Build model
print ("Build model...")
sequence_input = Input(shape=(time_steps,), dtype='float32')
print('Sequence input is:', sequence_input) # (batch_size, time_steps=500)
embedded_sequences = embedding_layer(sequence_input) 
print('Embedding layer is:', embedded_sequences) # (batch_size, time_steps=500, embedding_dim=25)

L = Bidirectional(GRU(n_x,
                      activation='tanh', 
                      dropout=0.2, 
                      recurrent_dropout=0.1, 
                      return_sequences=True,
                      kernel_initializer='he_uniform',
                      name='Pre-BiGRU'))(embedded_sequences)
print('Bi-GRU is:', L) # (batch_size, time_steps, units=32*2)

L = __attention3DBlock__(L) # (batch_size, time_steps=500, units=32*2)
print ('Attention layer is:', L)

L = GRU(n_x, 
        activation='tanh', 
        kernel_regularizer=regularizers.l2(0.001))(L)
print ('Post GRU is:', L)
L = Dense(2, 
          activation='softmax', 
          kernel_regularizer=regularizers.l2(0.001))(L)
print('Dense layer is:', L)

model = Model(inputs=sequence_input, outputs=L)

# Optimization and compile
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
print('Begin compiling...')
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])
model.summary()

# Begin training
model.fit(data_train, 
          Y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=2,
          validation_data=(data_val, Y_val))
score = model.evaluate(data_test, Y_test, batch_size=batch_size)
print ('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# Save model
print ('Saving model...')
model.save('GRU-orgAtt-20-cr-200d')