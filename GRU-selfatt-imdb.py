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
from keras.callbacks import *
import h5py
import matplotlib
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # Draw accuracy
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train_acc')
        # Draw loss
        plt.plot(iters, self.accuracy[loss_type],'g', label='train_loss')
        if loss_type == 'epoch':
            # Draw val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val_acc')
            # Draw val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='upper right')
        plt.show()


# Training Set
'''
X.shape = (m, seq_length)
Y.shape = (m, label)
label: [pos, neg], possibility of polar sentiment
'''
# Initialization
X = []
Y = None
MAX_LENGTH = 500
filepath1 = '/home/wenger/zhengxitian/atGRU/aclImdb/aclImdb/train/neg/'
filepath2 = '/home/wenger/zhengxitian/atGRU/aclImdb/aclImdb/train/pos/'

def __travDir__(filepath):
    label = 0
    if 'neg' in filepath:
        label = -1
    elif 'pos' in filepath:
        label = 1
    filelist = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        # Join direction and file path together
        child = os.path.join('%s%s' % (filepath, allDir))
        filelist.append(child)
    for n in filelist:
        __fill2DMat__(n, label)

def __fill2DMat__(filename, label):  
    global X, Y
    fopen = codecs.open(filename, 'r', 'utf-8', errors='ignore')
    if Y is None:
        if label == 1:
            Y = np.array([1,0])
        else:
            Y = np.array([0,1])
    else:
        if label == 1:
            Y = np.row_stack((Y, np.array([1,0])))
        else:
            Y = np.row_stack((Y, np.array([0,1])))
    # punctuation represented by 0?
    tmp = fopen.readline().strip().strip('\n')
    raw_word = []
    X.append(tmp)
# Dont need to seperate word now
#     for i in tmp:
#         raw_word.append(i)
#     # Fill zero in blank space
#     if len(raw_word) <= MAX_LENGTH:
#         for k in range(MAX_LENGTH-len(raw_word)):
#             raw_word.append(0)
#     if X is None:
#         X = [raw_word]
#     else:
#         X = X.append(raw_word)
    fopen.close()

# Running
__travDir__(filepath1)
__travDir__(filepath2)


# In[25]:


# Testing initialization
print('X.shape: ', len(X))
print('Y.shape: ', len(Y))
print(X[0])
print(Y[0])
#print(X[23064])
print(Y[23064])


# In[26]:


# Testing set
'''
X_test.shape = (n, seq_length)
Y_test.shape = (n, label)
label: [pos, neg], possibility of polar sentiment
'''
# Initialization
X_test = []
Y_test = None
MAX_LENGTH = 500
filepath3 = '/home/wenger/zhengxitian/atGRU/aclImdb/aclImdb/test/neg/'
filepath4 = '/home/wenger/zhengxitian/atGRU/aclImdb/aclImdb/test/pos/'

def __travDirTest__(filepath):
    label = 0
    if 'neg' in filepath:
        label = -1
    elif 'pos' in filepath:
        label = 1
    filelist = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        # Join direction and file path together
        child = os.path.join('%s%s' % (filepath, allDir))
        filelist.append(child)
    for n in filelist:
        __fill2DMatTest__(n, label)

def __fill2DMatTest__(filename, label):  
    global X_test, Y_test
    fopen = codecs.open(filename, 'r', 'utf-8', errors='ignore')
    if Y_test is None:
        if label == 1:
            Y_test = np.array([1,0])
        else:
            Y_test = np.array([0,1])
    else:
        if label == 1:
            Y_test = np.row_stack((Y_test, np.array([1,0])))
        else:
            Y_test = np.row_stack((Y_test, np.array([0,1])))
    # punctuation represented by 0?
    tmp = fopen.readline().strip().strip('\n')
    raw_word = []
    X_test.append(tmp)

__travDirTest__(filepath3)
__travDirTest__(filepath4)


# Testing initialization
print('X_test.shape: ', len(X_test))
print('Y_test.shape: ', len(Y_test))
print(X_test[0])
print(Y_test[0])
print(X_test[23183])
print(Y_test[23183])



# Parameters
EMBEDDING_DIM = 200
MAX_NUM_WORDS = 1200000

'''
GloVe Embeddings - Twitter - 2B tweets, 27B tokens, 1.2M vocab
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
Concat = X + X_test
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer.fit_on_texts(Concat)
sequences_train = tokenizer.texts_to_sequences(X)
sequences_test = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Auto filled with 0
data_train = pad_sequences(sequences_train, maxlen = MAX_LENGTH)
data_test = pad_sequences(sequences_test, maxlen = MAX_LENGTH)

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


# Testing embedding
print(embedding_matrix[100])
print(word_index.get('any'))
print(data_train[100])
print(data_test[100])



# Attention
'''
If SINGLE_ATTENTION_VECTOR = true, 
the attention vector is shared across the input_dimensions where the attention is applied.
'''
from keras.layers import multiply

SINGLE_ATTENTION_VECTOR = False
TIME_STEPS = 500

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



from keras import regularizers
from self_attention import SelfAttention
# Parameters
'''
n_x: hidden state size of Bi-GRU
regularization: L2 normalization
optimization: AdaGrad/Adam
time_steps = MAX_LENGTH = 500
'''
batch_size = 25
momentum = 0.9
l2_regularization = 0.001
learning_rate = 0.01
n_x = 32   
epochs = 15
time_steps = 500

# Build model
print ("Build model...")
sequence_input = Input(shape=(time_steps,), dtype='float32')
print('Sequence input is:', sequence_input) # (batch_size, time_steps=500)
embedded_sequences = embedding_layer(sequence_input) 
print('Embedding layer is:', embedded_sequences) # (batch_size, time_steps=500, embedding_dim=25)

# Self attention
self_att = SelfAttention(8,16)([embedded_sequences, embedded_sequences, embedded_sequences])


L = Bidirectional(GRU(n_x,
                      activation='tanh', 
                      dropout=0.2, 
                      recurrent_dropout=0.1, 
                      return_sequences=True,
                      kernel_initializer='he_uniform',
                      name='Pre-BiGRU'))(self_att)
print('Bi-GRU is:', L) # (batch_size, time_steps, units=32*2)

'''''''''''''''
Original attention:
''''''''''''
L = __attention3DBlock__(L) # (batch_size, time_steps=500, units=32*2)
print ('Attention layer is:', L)
'''''''''''''''

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

# Create instance history
# history = LossHistory()
# model = load_model('GRU-LSTM-5')

# Begin training
model.fit(data_train, 
          Y, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1) # print result per epoch 
        #  callbacks=[history])
score = model.evaluate(data_test, Y_test, batch_size=batch_size)
print ('Test score: ', score[0])
print('Test accuracy: ', score[1])


# Save model
#savepath = '/home/wenger/zhengxitian/atGRU/savedModels'
print ('Saving model...')
model.save('GRU-orgAtt-15-200d-IMDB')
