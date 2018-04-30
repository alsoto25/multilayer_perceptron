import utils as ut
import numpy as np


'''
Dropout
'''


class Network(object):
    def __init__(self, data, label, n_in, hidden_layer_sizes, n_out, rng=None):

        self.x = data
        self.y = label

        self.hidden_layers = []
        self.n_layers = len(hidden_layer_sizes)

        if rng is None:
            rng = np.random.RandomState(1234)

        assert self.n_layers > 0

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
                layer_data = self.hidden_layers[-1].output()

            # construct hidden_layer
            hidden_layer = HiddenLayer(data=layer_data,
                                       n_in=data_size,
                                       n_out=hidden_layer_sizes[i],
                                       rng=rng,
                                       activation=ut.ReLU)

            self.hidden_layers.append(hidden_layer)

            # layer for ouput using Logistic Regression (softmax)
            self.log_layer = LogisticRegression(data=self.hidden_layers[-1].output(),
                                                label=self.y,
                                                n_in=hidden_layer_sizes[-1],
                                                n_out=n_out)

    def feed_forward(self, dropout=True, p_dropout=0.5, rng=None):

        for epoch in range(ut.EPOCHS):
            dropout_masks = []  # create different masks in each training epoch

            # forward hidden_layers
            for i in range(self.n_layers):
                if i == 0:
                    layer_data = self.x

                layer_data = self.hidden_layers[i].forward(data=layer_data)

                if dropout:
                    mask = self.hidden_layers[i].dropout(data=layer_data, p=p_dropout, rng=rng)
                    layer_data *= mask

                    dropout_masks.append(mask)

            # forward & backward log_layer
            self.log_layer.train(data=layer_data)

            # backward hidden_layers
            for i in reversed(range(0, self.n_layers)):
                if i == self.n_layers - 1:
                    prev_layer = self.log_layer
                else:
                    prev_layer = self.hidden_layers[i + 1]

                self.hidden_layers[i].backward(prev_layer=prev_layer)

                if dropout:
                    self.hidden_layers[i].d_y *= dropout_masks[i]  # also mask here

    def predict(self, x, dropout=True, p_dropout=0.5):
        layer_data = x

        for i in range(self.n_layers):
            if dropout:
                self.hidden_layers[i].W = p_dropout * self.hidden_layers[i].W
                self.hidden_layers[i].b = p_dropout * self.hidden_layers[i].b

            layer_data = self.hidden_layers[i].output(data=layer_data)

        return self.log_layer.predict(layer_data)


'''
Hidden Layer
'''


class HiddenLayer(object):
    def __init__(self, data, n_in, n_out, W=None, b=None, rng=None, activation=ut.ReLU):

        if rng is None:
            rng = np.random.RandomState(1234)

        if W is None:
            W = np.random.randn(n_in, n_out)    # initialize W with normal standard distribution

        if b is None:
            b = np.zeros(n_out)  # initialize bias 0

        self.rng = rng
        self.x = data

        self.W = W
        self.b = b

        self.d_y = 0

        if activation == ut.tanh:
            self.prime_activation = ut.tanh_prime

        elif activation == ut.sigmoid:
            self.prime_activation = ut.sigmoid_prime

        elif activation == ut.ReLU:
            self.prime_activation = ut.ReLU_prime

        else:
            raise ValueError('activation function not supported.')

        self.activation = activation

    def output(self, data=None):
        if data is not None:
            self.x = data

        linear_output = np.dot(self.x, self.W) + self.b

        return (linear_output if self.activation is None
                else self.activation(linear_output))

    def sample_h_given_v(self, data=None):
        if data is not None:
            self.x = data

        v_mean = self.output()
        h_sample = self.rng.binomial(size=v_mean.shape,
                                     n=1,
                                     p=v_mean)
        return h_sample

    def forward(self, data=None):
        return self.output(data=data)

    def backward(self, prev_layer, lr=0.1, data=None):
        if data is not None:
            self.x = data

        d_y = self.prime_activation(prev_layer.x) * np.dot(prev_layer.d_y, prev_layer.W.T)

        self.W += lr * np.dot(self.x.T, d_y)
        self.b += lr * np.mean(d_y, axis=0)

        self.d_y = d_y

    def dropout(self, data, p, rng=None):
        if rng is None:
            rng = np.random.RandomState(123)

        mask = rng.binomial(size=data.shape,
                            n=1,
                            p=1 - p)  # p is the prob of dropping

        return mask


'''
Logistic Regression
'''


class LogisticRegression(object):
    def __init__(self, data, label, n_in, n_out, activation_function=ut.softmax, ):
        self.x = data
        self.y = label
        self.W = np.zeros((n_in, n_out))  # initialize W 0
        self.b = np.zeros(n_out)  # initialize bias 0
        self.output = np.zeros((data.shape[1], label.shape[0]))
        self.prime_output = np.zeros((label.shape[0], data.shape[1]))
        self.d_y = 0
        self.activation_function = activation_function

    def feed_forward_layer(self, data=None):
        if data is not None:
            self.x = data

        self.output = self.activation_function(np.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x

        self.W += ut.LEARNING_RATE * np.dot(self.x.T, d_y)
        self.b += ut.LEARNING_RATE * np.mean(d_y, axis=0)

        self.d_y = d_y

        # cost = self.negative_log_likelihood()
        # return cost

    def back_prop_layer(self):

    def cost(self):
        sigmoid_activation = self.activation_function(np.dot(self.x, self.W) + self.b)
        return ut.cross_entropy(sigmoid_activation, self.y)

    def predict(self, x):
        return self.activation_function(np.dot(x, self.W) + self.b)

    def output(self, x):
        return self.predict(x)


def test_dropout(n_epochs=5000, dropout=True, p_dropout=0.5):
    # XOR
    x = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

    y = np.array([[0, 1],
                     [1, 0],
                     [1, 0],
                     [0, 1]])

    rng = np.random.RandomState(123)

    # construct Dropout MLP
    classifier = Network(data=x, label=y,
                         n_in=2, hidden_layer_sizes=[10, 10], n_out=10,
                         rng=rng)

    # train
    classifier.feed_forward(dropout=dropout,
                            p_dropout=p_dropout, rng=rng)

    # test
    print(classifier.predict(x))


if __name__ == "__main__":
    test_dropout()