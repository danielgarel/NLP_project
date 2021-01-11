import tensorflow as tf
import random
import sklearn
from sklearn import metrics
import tensorflow.keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

random_SEED = 11
MAX_TRUNC_LEN = 30

'''Load the clean datasets'''
question1 = pd.read_csv('data/corpus/question1_sub_bal.clean.txt', sep="\n", header=None)
question2 = pd.read_csv('data/corpus/question2_sub_bal.clean.txt', sep="\n", header=None)
quora_split = pd.read_csv('data/quora_split_balanced_subset.txt', sep='\t', header = None)
question1.columns = ['question']
question2.columns = ['question']
quora_split.columns = ['index', 'label','train_test']


'''Load the train-test split'''
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('data/' + 'quora_split_balanced_subset' + '.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[2].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[2].find('train') != -1:
            doc_train_list.append(line.strip())

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.Random(random_SEED).shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.Random(random_SEED).shuffle(test_ids)

'''Apply Keras Tokenizer'''
MAX_NB_WORDS = 200000
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(list(question1['question'].values.astype(str))+list(question2['question'].values.astype(str)))


X_train_q1 = tokenizer.texts_to_sequences(question1.loc[train_ids]['question'].values.astype(str))
X_train_q1 = pad_sequences(X_train_q1, maxlen=MAX_TRUNC_LEN, padding='post')

X_train_q2 = tokenizer.texts_to_sequences(question2.loc[train_ids]['question'].values.astype(str))
X_train_q2 = pad_sequences(X_train_q2, maxlen=MAX_TRUNC_LEN, padding='post')

X_test_q1 = tokenizer.texts_to_sequences(question1.loc[test_ids]['question'].values.astype(str))
X_test_q1 = pad_sequences(X_test_q1, maxlen=MAX_TRUNC_LEN, padding='post')

X_test_q2 = tokenizer.texts_to_sequences(question2.loc[test_ids]['question'].values.astype(str))
X_test_q2 = pad_sequences(X_test_q2, maxlen=MAX_TRUNC_LEN, padding='post')

word_index =tokenizer.word_index

Y_train = quora_split.loc[train_ids]['label'].values
Y_test = quora_split.loc[test_ids]['label'].values
print(Y_test)

# load pre-trained word embeddings
word_embeddings_dim = 300
embedding_index = {}

with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_index[word] = vectors
    f.close()

embedding_matrix = np.random.random((len(word_index)+1, 300))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

'''LSTM'''

# Model for Q1
model_q1 = tf.keras.Sequential()
model_q1.add(Embedding(input_dim = len(word_index)+1,
                       output_dim = 300,
                      weights = [embedding_matrix],
                      input_length = MAX_TRUNC_LEN))
model_q1.add(LSTM(128, activation = 'tanh', return_sequences = True))
model_q1.add(Dropout(0.2))
model_q1.add(LSTM(128, return_sequences = True))
model_q1.add(LSTM(128))
model_q1.add(Dense(80, activation = 'tanh'))
model_q1.add(Dense(30, activation = 'sigmoid'))

# Model for Q2
model_q2 = tf.keras.Sequential()
model_q2.add(Embedding(input_dim = len(word_index)+1,
                       output_dim = 300,
                      weights = [embedding_matrix],
                      input_length = MAX_TRUNC_LEN))
model_q2.add(LSTM(128, activation = 'tanh', return_sequences = True))
model_q2.add(Dropout(0.2))
model_q2.add(LSTM(128, return_sequences = True))
model_q2.add(LSTM(128))
model_q2.add(Dense(80, activation = 'tanh'))
model_q2.add(Dense(30, activation = 'sigmoid'))

# Merging the output of the two models,i.e, model_q1 and model_q2

mergedOut = Multiply()([model_q1.output, model_q2.output])

mergedOut = Flatten()(mergedOut)
mergedOut = Dense(100, activation = 'relu')(mergedOut)
mergedOut = Dropout(0.2)(mergedOut)
mergedOut = Dense(50, activation = 'relu')(mergedOut)
mergedOut = Dropout(0.2)(mergedOut)
mergedOut = Dense(2, activation = 'softmax')(mergedOut)

file_path = "save.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_accuracy", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]

new_model = Model([model_q1.input, model_q2.input], mergedOut)
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
new_model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'], )
history = new_model.fit([X_train_q1,X_train_q2],Y_train, callbacks=callbacks_list, batch_size = 500, epochs = 15, validation_split = 0.1)
new_model.load_weights(file_path)

y_pred = new_model.predict([X_test_q1, X_test_q2], batch_size=500, verbose=1)[:, 0]
y_pred = (y_pred<0.5).astype(np.int8)
accuracy = accuracy_score(Y_test,y_pred)
print(accuracy)

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(Y_test, y_pred, digits=4))
