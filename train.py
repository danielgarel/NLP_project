from __future__ import division
from __future__ import print_function

import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn import metrics
import pickle as pkl

from utils import *
from models import GNN, MLP, GNN_sim

# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('dataset', 'mr', 'Dataset string.')  # 'mr','ohsumed','R8','R52'
flags.DEFINE_string('model', 'gnn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 4096, 'Size of batches per epoch.') 
flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
flags.DEFINE_integer('hidden', 96, 'Number of units in hidden layer.') # 32, 64, 96, 128
flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.') # 5e-4
flags.DEFINE_integer('early_stopping', -1, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # Not used

# Load data
train_adj1, train_feature1, train_y1, val_adj1, val_feature1, val_y1, test_adj1, test_feature1, test_y1 = load_data('question1')
train_adj2, train_feature2, train_y2, val_adj2, val_feature2, val_y2, test_adj2, test_feature2, test_y2 = load_data('question2')

# concat_ratio = FLAGS.batch_size/2
# def concat_(train_test_y, half_batch =concat_ratio, input1, input2):
#     indices = np.arange(0, len(train_test_y))
#     for start in range(0, len(train_test_y), half_batch):
#         end = start + half_batch
#         idx = indices[start:end]
#         output = np.concatenate(input1[idx], input2[idx])
#     return output
# train_adj = concat_(train_test_y = train_y1, half_batch = concat_ratio, input1 = train_adj1, input2 = train_adj2)
# train_feature = concat_(train_test_y = train_y1, half_batch = concat_ratio, input1 = train_feature1, input2 = train_feature2)
# val_adj = concat_(train_test_y = val_adj1, half_batch = concat_ratio, input1 = val_adj1, input2 = val_adj2)
# val_feature = concat_(train_test_y = val_adj1, half_batch = concat_ratio, input1 = val_feature1, input2 = val_feature2)
# test_adj = concat_(train_test_y = val_adj1, half_batch = concat_ratio, input1 = val_feature1, input2 = val_feature2)

# Some preprocessing
print('loading training set')
m = max([a.shape[0] for a in train_adj1] + [a.shape[0] for a in train_adj2])
train_adj1, train_mask1 = preprocess_adj(train_adj1, m)
train_adj2, train_mask2 = preprocess_adj(train_adj2, m)

m = max([len(f) for f in train_feature1] + [len(f) for f in train_feature2])
train_feature1 = preprocess_features(train_feature1, m)
train_feature2 = preprocess_features(train_feature2, m)
print(train_adj1.shape)
print(train_mask1.shape)
print(train_feature1.shape)
print(train_adj2.shape)
print(train_mask2.shape)
print(train_feature2.shape)
print('loading validation set')
m = max([a.shape[0] for a in val_adj1] + [a.shape[0] for a in val_adj2])
val_adj1, val_mask1 = preprocess_adj(val_adj1, m)
val_adj2, val_mask2 = preprocess_adj(val_adj2, m)
print(val_mask1.shape)
print(val_adj1.shape)

m = max([len(f) for f in val_feature1] + [len(f) for f in val_feature2])
val_feature1 = preprocess_features(val_feature1, m)
val_feature2 = preprocess_features(val_feature2, m)
print('loading test set')
m = max([a.shape[0] for a in test_adj1] + [a.shape[0] for a in test_adj2])
test_adj1, test_mask1 = preprocess_adj(test_adj1, m)
test_adj2, test_mask2 = preprocess_adj(test_adj2, m)

m = max([len(f) for f in test_feature1] + [len(f) for f in test_feature2])
test_feature1 = preprocess_features(test_feature1, m)
test_feature2 = preprocess_features(test_feature2, m)

if FLAGS.model == 'gnn': #TODO Can change if we want to make it more precise to have the gnn_sim only
    # support = [preprocess_adj(adj)]
    # num_supports = 1
    model_func = GNN_sim
elif FLAGS.model == 'gcn_cheby': # not used
    # support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GNN_sim
elif FLAGS.model == 'dense': # not used
    # support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# Define placeholders
placeholders = {
    'support': tf.placeholder(tf.float32, shape=(None, None, None)),
    'features': tf.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
    'mask': tf.placeholder(tf.float32, shape=(None, None, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, train_y1.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}


# label smoothing
# label_smoothing = 0.1
# num_classes = y_train.shape[1]
# y_train = (1.0 - label_smoothing) * y_train + label_smoothing / num_classes


# Create model
model = model_func(placeholders, input_dim=FLAGS.input_dim, logging=True)

# Initialize session
sess = tf.Session()

# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('logs/', sess.graph)

# Define model evaluation function
def evaluate(features, support, mask, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
best_val = 0
best_epoch = 0
best_acc = 0
best_cost = 0
test_doc_embeddings = None
preds = None
labels = None

print('train start...')
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
        
    # Training step
    indices = np.arange(0, len(train_y1)) #TODO confirm that len(train_y1 = train_y2 = train_y)
    np.random.shuffle(indices)
    
    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y1), FLAGS.batch_size):
        end = start + FLAGS.batch_size
        idx = indices[start:end]
        # Construct feed dictionary
        concat_ratio = FLAGS.batch_size/2
        def concat_(train_test_y, concat_ratio, input1, input2):
            indices = np.arange(0, len(train_test_y))
            for start in range(0, len(train_test_y), half_batch):
                end = start + half_batch
                idx = indices[start:end]
                output = np.concatenate(input1[idx], input2[idx])
            return output
        #train_feature = concat_(train_y1, concat_ratio, train_feature1, train_feature2)
        #train_adj = concat_(train_y1, concat_ratio, train_adj1, train_adj2)
        #train_mask = concat_(train_y1, concat_ratio, train_mask1, train_mask2)
        #train_y =concat_(train_y1, concat_ratio, train_y1, train_y2)
        feed_dict = construct_feed_dict(np.concatenate((train_feature1[idx], train_feature2[idx]), axis=0), np.concatenate((train_adj1[idx], train_adj2[idx]), axis=0), np.concatenate((train_mask1[idx], train_mask2[idx]), axis=0), train_y1[idx], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        train_loss += outs[1]*len(idx)
        train_acc += outs[2]*len(idx)
    train_loss /= len(train_y1)
    train_acc /= len(train_y1)

    # Validation
    val_cost, val_acc, val_duration, _, _, _ = evaluate(np.concatenate((val_feature1, val_feature2), axis=0), np.concatenate((val_adj1, val_adj2), axis=0), np.concatenate((val_mask1, val_mask2), axis=0), val_y1, placeholders)
    cost_val.append(val_cost)
    
    # Test
    test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(np.concatenate((test_feature1, test_feature2), axis=0), np.concatenate((test_adj1, test_adj2), axis=0), np.concatenate((test_mask1, test_mask2), axis=0), test_y1, placeholders)

    if val_acc >= best_val:
        best_val = val_acc
        best_epoch = epoch
        best_acc = test_acc
        best_cost = test_cost
        test_doc_embeddings = embeddings
        preds = pred

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
          "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(test_acc), 
          "time=", "{:.5f}".format(time.time() - t))

    if FLAGS.early_stopping > 0 and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Best results
print('Best epoch:', best_epoch)
print("Test set results:", "cost=", "{:.5f}".format(best_cost),
      "accuracy=", "{:.5f}".format(best_acc))

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(labels, preds, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))

'''
# For visualization
doc_vectors = []
for i in range(len(test_doc_embeddings)):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append(str(np.argmax(test_y[i])) + ' ' + doc_vector_str)

doc_embeddings_str = '\n'.join(doc_vectors)
with open('data/' + FLAGS.dataset + '_doc_vectors.txt', 'w'):
    f.write(doc_embeddings_str)
'''
