from layers import *
from metrics import *

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.embeddings = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations = [self.inputs]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embeddings = self.activations[-2]
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(ReadoutLayer(input_dim=FLAGS.hidden,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GNN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.mask = placeholders['mask']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        print('build...')
        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if 'weights' in var.name or 'bias' in var.name:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        
        self.layers.append(GraphLayer(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden,
                                      placeholders=self.placeholders,
                                      act=tf.tanh,
                                      sparse_inputs=False,
                                      dropout=True,
                                      steps=FLAGS.steps,
                                      logging=self.logging))

        self.layers.append(ReadoutLayer(input_dim=FLAGS.hidden,
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=tf.tanh,
                                        sparse_inputs=False,
                                        dropout=True,
                                        logging=self.logging))
        
    def predict(self):
        return tf.nn.softmax(self.outputs)

class GNN_sim(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GNN_sim, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.mask = placeholders['mask']
        self.placeholders = placeholders
        self.classifier = None
        self.classification_features = None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        print(self.output_dim)
        print('build...')
        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if 'weights' in var.name or 'bias' in var.name:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        # Latent representation of questions
        self.activations = [self.inputs]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embeddings = self.activations[-1]
        print(self.embeddings)
        
        # Recreate pairs
        questions1, questions2 = tf.split(self.embeddings, num_or_size_splits=2, axis=0)

        # Create feature for classification
        sub = tf.math.abs(tf.math.subtract(questions1, questions2))
        mul = tf.math.multiply(questions1, questions2)
        #dot = tf.reduce_sum( mul, 1, keep_dims=True )
        #l2 = tf.reduce_sum(tf.multiply( sub, sub ), 1, keep_dims=True)

        self.classification_features = tf.concat([sub, mul], 1)
        print(self.classification_features)

        self.outputs = self.classifier(self.classification_features)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        
        self.layers.append(GraphLayer(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden,
                                      placeholders=self.placeholders,
                                      act=tf.tanh,
                                      sparse_inputs=False,
                                      dropout=True,
                                      steps=FLAGS.steps,
                                      logging=self.logging))

        self.layers.append(ReadoutLayer_(input_dim=FLAGS.hidden,
                                        placeholders=self.placeholders,
                                        act=tf.tanh,
                                        sparse_inputs=False,
                                        dropout=True,
                                        logging=self.logging))

        self.classifier = Dense_(input_dim=2*FLAGS.hidden,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=None,
                                 bias=True,
                                 dropout=False,
                                 sparse_inputs=False,
                                 logging=self.logging)
        
    def predict(self):
        return tf.nn.softmax(self.outputs)

class DNN_sim(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(DNN_sim, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.mask = placeholders['mask']
        self.placeholders = placeholders
        self.classifier = None
        self.classification_features = None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        print(self.output_dim)
        print('build...')
        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if 'weights' in var.name or 'bias' in var.name:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])
        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Latent representation of questions
        self.activations = [self.inputs]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embeddings = self.activations[-1]
        
        # Recreate pairs
        questions1, questions2 = tf.split(self.embeddings, num_or_size_splits=2, axis=0)

        # Create feature for classification
        sub = tf.math.abs(tf.math.subtract(questions1, questions2))
        mul = tf.math.multiply(questions1, questions2)
        #dot = tf.reduce_sum( mul, 1, keep_dims=True )
        #l2 = tf.reduce_sum(tf.multiply( sub, sub ), 1, keep_dims=True)

        self.classification_features = tf.concat([sub, mul], 1)

        self.outputs = self.classifier(self.classification_features)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        
        self.layers.append(Dense(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden,
                                      placeholders=self.placeholders,
                                      act=tf.tanh,
                                      sparse_inputs=False,
                                      dropout=True,
                                      logging=self.logging))

        self.layers.append(ReadoutLayer_(input_dim=FLAGS.hidden,
                                        placeholders=self.placeholders,
                                        act=tf.tanh,
                                        sparse_inputs=False,
                                        dropout=True,
                                        logging=self.logging))

        self.classifier = Dense_(input_dim=2*FLAGS.hidden,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=None,
                                 bias=True,
                                 dropout=False,
                                 sparse_inputs=False,
                                 logging=self.logging)
        
    def predict(self):
        return tf.nn.softmax(self.outputs)

