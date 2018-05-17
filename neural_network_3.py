import math
import utils as ut
import numpy as np
import scipy.misc as smp


class Network(object):
    def __init__(self, data, labels, n_in, hidden_layer_sizes, n_out, test_data, test_labels, rng=None, dropout=True):
        self.n_in = n_in
        self.n_out = n_out
        self.epoch = 0
        self.hidden_layer_sizes = hidden_layer_sizes
        self.x = data / 255
        self.y = labels
        self.test_data = test_data / 255
        self.test_labels = test_labels
        self.dropout = dropout

        self.accs = []
        self.costs = []
        self.hidden_layers = []
        self.dropout_masks = []
        self.n_layers = len(hidden_layer_sizes)

        if rng is None:
            rng = np.random.RandomState(1234)

        assert self.n_layers >= 0

        # construct multi-layer
        for i in range(self.n_layers):

            # layer_size
            if i == 0:
                data_size = n_in
            else:
                data_size = hidden_layer_sizes[i - 1]

            # layer_data
            if i == 0:
                layer_data = self.x

            else:
                layer_data = self.hidden_layers[-1].output

            # construct hidden_layer
            hidden_layer = HiddenLayer(data=layer_data,
                                       n_in=data_size,
                                       n_out=hidden_layer_sizes[i],
                                       rng=rng)

            self.hidden_layers.append(hidden_layer)

        if self.n_layers != 0:
            self.log_layer = LogisticRegression(data=self.hidden_layers[-1].output,
                                                label=self.y,
                                                n_in=hidden_layer_sizes[-1],
                                                n_out=n_out)
        else:
            self.log_layer = LogisticRegression(data=self.x,
                                                label=self.y,
                                                n_in=n_in,
                                                n_out=n_out)

    def train(self, batch=True):
        for j in range(ut.EPOCHS):
            self.epoch = j

            print('------------------------------- E P O C H   # ' + str(j) + ' -------------------------------')
            data = np.asarray([(x, y) for x, y in zip(self.x, self.y)])
            np.random.shuffle(data)

            if batch and ut.BATCH_SIZE <= len(self.x):
                batch_size_range = int(len(self.x) / ut.BATCH_SIZE) - 1
            else:
                batch_size_range = 1
            for i in range(batch_size_range):
                # print("------------------ Batch #" + str(i + 1) + " of " + str(batch_size_range) +
                #       '  /////////  Epoch #' + str(j + 1))

                dropout = i == 0

                cost = self.feed_forward_batch(
                    data=np.asarray([x for (x, y) in data[(i * ut.BATCH_SIZE):((i + 1) * ut.BATCH_SIZE)]]),
                    labels=np.asarray([y for (x, y) in data[(i * ut.BATCH_SIZE):((i + 1) * ut.BATCH_SIZE)]]),
                    dropout=dropout)

                self.back_prop_batch()
                self.update_batch()

                # print('Cost: ' + str(cost))
                # if batch:
                #     print('Batch Accuracy: ' + str(self.get_results(np.asarray([y for (x, y) in data[(i * ut.BATCH_SIZE):((i + 1) * ut.BATCH_SIZE)]]))) + '%')

            acc, test_cost = self.test(self.test_data, self.test_labels)
            print('Accuracy: ' + str(acc) + '%')
            print('Test Cost: ' + str(test_cost))

            is_max = self.accs == [] or acc > np.max(np.asarray(self.accs))
            self.costs.append(test_cost)
            self.accs.append(acc)
            if j % ut.EPOCHS_DROP == 0 or is_max:
                lay = ''
                for x in self.hidden_layer_sizes:
                    lay += str(x) + '_'

                self.save('nn_' + lay +
                          'ep_' + str(j + 1) +
                          '_lr_' + str(ut.LEARNING_RATE) +
                          '_dp_' + str(ut.DROPOUT_PERCENTAGE) +
                          '_bs_' + str(ut.BATCH_SIZE) +
                          '_acc_' + str(acc))

    def feed_forward_batch(self, data=None, labels=None, rng=None, dropout=True):
        self.dropout_masks = []

        if labels is None:
            labels = self.y

        if data is not None:
            layer_data = data
        else:
            layer_data = self.x

        for i in range(self.n_layers):
            layer_data = self.hidden_layers[i].feed_forward(data=layer_data)

            if self.dropout:
                mask = self.hidden_layers[i].dropout(data=layer_data, p=ut.DROPOUT_PERCENTAGE, rng=rng)
                layer_data *= mask

                self.dropout_masks.append(mask)

        return self.log_layer.feed_forward_layer(data=layer_data, labels=labels)

    def back_prop_batch(self):
        self.log_layer.back_prop_layer()

        for i in reversed(range(self.n_layers)):
            if i == self.n_layers - 1:
                prev_layer = self.log_layer
            else:
                prev_layer = self.hidden_layers[i + 1]

            self.hidden_layers[i].back_prop(prev_layer=prev_layer)

            if self.dropout:
                self.hidden_layers[i].prime_output *= self.dropout_masks[i]  # also mask here

    def update_batch(self):
        for i in range(self.n_layers):
            self.hidden_layers[i].update(self.epoch)

        self.log_layer.update(self.epoch)

    def predict(self, x):
        layer_data = x

        for i in range(self.n_layers):
            # if self.dropout:
            #     self.hidden_layers[i].W = ut.DROPOUT_PERCENTAGE * self.hidden_layers[i].W
            #     self.hidden_layers[i].b = ut.DROPOUT_PERCENTAGE * self.hidden_layers[i].b

            layer_data = self.hidden_layers[i].feed_forward(data=layer_data)

        return self.log_layer.predict(layer_data)

    def test(self, data, labels):
        cost_list = []
        result_list = []

        new_data = np.asarray([(x, y) for x, y in zip(data / np.max(data), labels)])
        np.random.shuffle(new_data)

        if ut.BATCH_SIZE <= len(new_data):
            batch_size_range = int(len(new_data) / ut.BATCH_SIZE) - 1
        else:
            batch_size_range = 1

        for j in range(batch_size_range):
            layer_data = np.asarray([x for (x, y) in new_data[(j * ut.BATCH_SIZE):((j + 1) * ut.BATCH_SIZE)]])
            new_labels = np.asarray([y for (x, y) in new_data[(j * ut.BATCH_SIZE):((j + 1) * ut.BATCH_SIZE)]])

            for i in range(self.n_layers):
                layer_data = self.hidden_layers[i].feed_forward(data=layer_data)

            cost_list.append(self.log_layer.feed_forward_layer(data=layer_data, labels=new_labels))
            result_list.append(self.get_results(new_labels))

        return np.mean(np.asarray(result_list)), np.mean(np.asarray(cost_list))

    def get_results(self, labels):
        results = [(np.argmax(x), y) for x, y in zip(self.log_layer.output, labels)]
        return sum(int(x == y) for (x, y) in results) * 100 / labels.size

    def save(self, filename):
        data = {}

        data['general_info'] = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'n_in': self.n_in,
            'n_out': self.n_out,
            'accs': self.accs,
            'costs': self.costs
        }

        for i in range(self.n_layers):
            data['hidden_layer_' + str(i)] = {
                'activation': self.hidden_layers[i].activation.__name__,
                'w': self.hidden_layers[i].W,
                'b': self.hidden_layers[i].b
            }

        data['log_layer'] = {
                'activation': self.log_layer.activation_function.__name__,
                'w': self.log_layer.W,
                'b': self.log_layer.b
            }

        ut.pickle(ut.PICKLES_DIR + filename.replace('.', ''), data)

    def load(self, filename):
        data = ut.unpickle(ut.PICKLES_DIR + filename)

        self.n_in = data['general_info']['n_in']
        self.n_out = data['general_info']['n_out']
        self.costs = data['general_info']['costs']
        self.accs = data['general_info']['accs']
        self.hidden_layer_sizes = data['general_info']['hidden_layer_sizes']

        self.n_layers = len(self.hidden_layer_sizes)

        self.hidden_layers = []
        self.dropout_masks = []

        assert self.n_layers >= 0

        # construct multi-layer
        for i in range(self.n_layers):

            # layer_size
            if i == 0:
                data_size = self.n_in
            else:
                data_size = self.hidden_layer_sizes[i - 1]

            # layer_data
            if i == 0:
                layer_data = self.x

            else:
                layer_data = self.hidden_layers[-1].output

            # construct hidden_layer
            hidden_layer = HiddenLayer(data=layer_data,
                                       n_in=data_size,
                                       n_out=self.hidden_layer_sizes[i],
                                       activation=getattr(ut, data['hidden_layer_' + str(i)]['activation']))

            hidden_layer.set_wb(data['hidden_layer_' + str(i)]['w'], data['hidden_layer_' + str(i)]['b'])

            self.hidden_layers.append(hidden_layer)

        if self.n_layers != 0:
            self.log_layer = LogisticRegression(data=self.hidden_layers[-1].output,
                                                label=self.y,
                                                n_in=self.hidden_layer_sizes[-1],
                                                n_out=self.n_out,
                                                activation_function=getattr(ut,
                                                                            data['log_layer']['activation']))
        else:
            self.log_layer = LogisticRegression(data=self.x,
                                                label=self.y,
                                                n_in=self.n_in,
                                                n_out=self.n_out,
                                                activation_function=getattr(ut,
                                                                            data['log_layer']['activation']))

        self.log_layer.set_wb(data['log_layer']['w'], data['log_layer']['b'])

    def print_img(self, layer, neuron, filename='tmp_test_img'):
        img_mat = (layer.W.T[neuron].reshape(28, 28) * 255)
        img = smp.toimage(img_mat)
        img.save('./tmp/' + filename + '.bmp')

'''
Hidden Layer
'''


class HiddenLayer(object):
    def __init__(self, data, n_in, n_out, W=None, b=None, rng=None, activation=ut.ReLU):

        if rng is None:
            rng = np.random.RandomState(1234)

        if W is None:
            # W = np.random.normal(ut.MU, ut.SIGMA, (n_in, n_out))
            W = np.random.randn(n_in, n_out) / np.sqrt(n_in)

        if b is None:
            # b = np.random.normal(ut.MU, ut.SIGMA, n_out)
            b = np.random.randn(n_out) / np.sqrt(n_in)


        self.rng = rng
        self.x = data

        self.W = W
        self.b = b

        self.output = np.zeros(n_out)
        self.prime_output = np.zeros((n_out, n_in))

        if activation == ut.tanh:
            self.prime_activation = ut.tanh_prime

        elif activation == ut.sigmoid:
            self.prime_activation = ut.sigmoid_prime

        elif activation == ut.ReLU:
            self.prime_activation = ut.ReLU_prime

        else:
            raise ValueError('activation function not supported.')

        self.activation = activation

    def feed_forward(self, data=None):
        if data is not None:
            self.x = data

        self.output = self.activation(np.dot(self.x, self.W) + self.b)
        return self.output

    def back_prop(self, prev_layer, data=None):
        if data is not None:
            self.x = data

        self.prime_output = self.prime_activation(prev_layer.x) * np.dot(prev_layer.prime_output, prev_layer.W.T)

    def update(self, epochs):
        lr_decay = ut.LEARNING_RATE * math.pow(ut.LEARNING_RATE_DECAY,
                                               math.floor(epochs / ut.EPOCHS_DROP))
        self.W += lr_decay * np.dot(self.x.T, self.prime_output)
        self.b += lr_decay * np.mean(self.prime_output, axis=0)

    def dropout(self, data, p, rng=None):
        if rng is None:
            rng = np.random.RandomState(123)

        mask = rng.binomial(size=data.shape[1],
                            n=1,
                            p=1 - p)  # p is the prob of dropping

        return mask

    def set_wb(self, w, b):
        self.W = w
        self.b = b


'''
Logistic Regression
'''


class LogisticRegression(object):
    def __init__(self, data, label, n_in, n_out, activation_function=ut.softmax):
        self.x = data
        self.y = label

        # self.W = np.random.normal(ut.MU, ut.SIGMA, (n_in, n_out))
        self.W = np.random.randn(n_in, n_out) / np.sqrt(n_in)
        # self.b = np.random.normal(ut.MU, ut.SIGMA, n_out)
        self.b = np.random.randn(n_out) / np.sqrt(n_in)

        self.output = np.zeros(n_out)
        self.prime_output = np.zeros((n_out, n_in))
        self.activation_function = activation_function

    def feed_forward_layer(self, data=None, labels=None):
        if data is not None:
            self.x = data
        if labels is not None:
            self.y = labels

        self.output = self.activation_function(np.dot(self.x, self.W) + self.b)

        return self.cost()

    def back_prop_layer(self):
        self.prime_output = ut.cross_entropy_prime(self.output, self.y)

    def update(self, epochs):
        lr_decay = ut.LEARNING_RATE * math.pow(ut.LEARNING_RATE_DECAY,
                                               math.floor(epochs / ut.EPOCHS_DROP))
        self.W += lr_decay * np.dot(self.x.T, self.prime_output)
        self.b += lr_decay * np.mean(self.prime_output, axis=0)

    def cost(self):
        return ut.cross_entropy(self.output, self.y)

    def predict(self, x):
        return np.argmax(self.activation_function(np.dot(x, self.W) + self.b))

    def set_wb(self, w, b):
        self.W = w
        self.b = b
