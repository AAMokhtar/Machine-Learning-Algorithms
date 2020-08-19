import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# a multilayer perceptron architecture based on Andrew ng's courses. fitted on the
# Ghouls, Goblins, and Ghosts dataset.
# the network uses mini-batch gradient descent optimized with the Adam algorithm.
# it also employs learning rate decay for better convergence in later epochs.

def train(X, Y, layer_dims, activations,learning_rate=0.0075,lambd=0.0,
          epochs=1, mini_batch_size=32, beta1=0.9, beta2=0.999,epsilon=1e-8,
          optimizer="gd", lr_decay=0.0005, report=False,save=False):
    # list of costs (for debugging)
    # costs = []

    # seed for epoch randomization
    seed = 1

    # initialize weights and biases
    params = params_init(X, layer_dims)

    # optimizer initialization
    if optimizer == "adam":
        t = 1
        Mt, Vt = initialize_adam(params)

    for i in range(epochs):

        # randomize mini batches
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        seed += 1

        # update the learning rate
        learning_rate_new = learning_rate / (1 + i * lr_decay)

        for mini_batch in mini_batches:

            # get the current minibatch
            (mX, mY) = mini_batch

            # forward propagation
            AL, cache = forward_prop(mX, params, activations)

            # get cost (for debugging)
            if lambd != 0.0:
                curCost = cost_L2(AL, mY, params, lambd)
            else:
                curCost = cost(AL, mY)

            # back propagation
            if lambd != 0.0: #L2 regularization coefficient (lambda)
                grads = back_prop_L2(params, cache, activations, mY, lambd)
            else:
                grads = back_prop(params, cache, activations, mY)

            # update parameters
            if optimizer == "adam":
                params, Mt, Vt = update_parameters_Adam(params, grads, Mt, Vt, t, learning_rate_new, beta1,
                                                               beta2, epsilon)
                t += 1
            else:
                params = update_params(params, grads, learning_rate_new)

        # print cost every 100 epochs (for debugging)
        # if report and i % 100 == 0: print("cost after %d epochs: %f" % (i, curCost))
        # if report and i % 100 == 0:
        #     costs.append(curCost)

    # save the learned parameters
    if save:
        file = open("learned_Parameters","wb")
        pickle.dump(params,file)
        file.close()

    # plot the cost (for debugging)
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('epochs')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

# ===========================normalization functions=================================
def linear_scaling(X):
    min = np.amin(X, axis=1, keepdims=True)
    max = np.amax(X, axis=1, keepdims=True)

    return (X - min) / (max - min)

# ===========================initialization functions=================================
#weights and biases
def params_init(X, layer_dims):
    # L = number of layers (not counting input)
    L = len(layer_dims)

    # insert the input layer size to make things easier
    layer_dims.insert(0, X.shape[0])
    weights = {}
    biases = {}
    np.random.seed(1)

    # - randomize weights and multiply them by np.sqrt(1./layer_dims[i - 1])
    # to avoid vanishing/exploding gradients.
    # - initialize biases as 0s
    for i in range(1, L + 1):
        weights["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(1./layer_dims[i - 1])
        biases["b" + str(i)] = np.zeros((layer_dims[i], 1))

    return {"weights": weights, "biases": biases}

# Adam's momentum, RMS terms
def initialize_adam(params):
    W = params["weights"]
    b = params["biases"]
    L = len(W)
    Mt = {}
    Vt = {}

    # for each layer
    for i in range(L):
       Mt["dW" + str(i + 1)] = np.zeros(W["W" + str(i + 1)].shape)
       Mt["db" + str(i + 1)] = np.zeros(b["b" + str(i + 1)].shape)

       Vt["dW" + str(i + 1)] = np.zeros(W["W" + str(i + 1)].shape)
       Vt["db" + str(i + 1)] = np.zeros(b["b" + str(i + 1)].shape)

    return Mt,Vt

# ===========================forward propagation functions=================================
#normal forward prop
def forward_prop(X, params, activations):
    L = len(params["weights"])
    W = params["weights"]
    b = params["biases"]
    cache = {"A0": X}

    # each perceptron calculates a linear function (WX + b)
    # followed by an activation function to introduce non-linearity
    # to the network
    for i in range(1, L + 1):
        cache["Z" + str(i)] = np.dot(W["W" + str(i)], cache["A" + str(i - 1)]) + b["b" + str(i)]

        if activations[i - 1] == "relu":
            cache["A" + str(i)] = relu(cache["Z" + str(i)])
        elif activations[i - 1] == "swish":
            cache["A" + str(i)] = swish(cache["Z" + str(i)])
        elif activations[i - 1] == "sigmoid":
            cache["A" + str(i)] = sigmoid(cache["Z" + str(i)])
        elif activations[i - 1] == "tanh":
            cache["A" + str(i)] = tanh(cache["Z" + str(i)])
        elif activations[i - 1] == "softmax":
            cache["A" + str(i)] = softmax(cache["Z" + str(i)])

    return cache["A" + str(L)], cache


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def swish(Z):
    return Z * sigmoid(Z)

def tanh(Z):
    return np.tanh(Z)

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z = np.exp(Z - np.max(Z,axis=0,keepdims=True))
    sumAL = np.sum(Z, axis=0)
    return Z / sumAL

# ===========================cost calculation functions=================================
# cross_entropy loss
def cross_entropy(AL,Y):
    return - np.sum(np.log(AL) * Y,axis=0)

#normal cost
def cost(AL, Y):
    m = Y.shape[1]
    return np.sum(cross_entropy(AL, Y)) / m

# cost with L2 loss
def cost_L2(AL, Y,params,lambd):
    W = params["weights"]
    m = Y.shape[1]
    L2 = 0
    for cur in W:
        L2 = np.squeeze(L2 + np.sum(np.square(W[cur])))
    L2 = (lambd / (2 * m)) * L2


    return cost(AL,Y) + L2

# ===========================back propagation functions=================================
# normal back propagation
def back_prop(params, cache, activations, Y):
    W = params["weights"]
    L = len(W)
    m = Y.shape[1]

    grads = {}

    for i in reversed(range(L)):
        # ==========dZ===========
        if activations[i] == "relu":
            dZ = dA * relu_back(cache["Z" + str(i + 1)])
        elif activations[i] == "swish":
            dZ = dA * swish_back(cache["Z" + str(i + 1)])
        elif activations[i] == "sigmoid":
            dZ = dA * sigmoid_back(cache["Z" + str(i + 1)])
        elif activations[i] == "tanh":
            dZ = dA * tanh_back(cache["Z" + str(i + 1)])
        elif activations[i] == "softmax":
            dZ = cache["A" + str(L)] - Y
        # ==========dW===========
        dW = (1 / m) * np.dot(dZ, cache["A" + str(i)].T)
        # ==========db===========
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # ==========dA===========
        dA = np.dot(W["W" + str(i + 1)].T, dZ)
        # ======gradients========
        grads["dW" + str(i + 1)] = dW
        grads["db" + str(i + 1)] = db
    return grads

# L2 loss back propagation
def back_prop_L2(params, cache, activations, Y,lambd):
    W = params["weights"]
    L = len(W)
    m = Y.shape[1]

    grads = {}

    for i in reversed(range(L)):
        # ==========dZ===========
        if activations[i] == "relu":
            dZ = dA * relu_back(cache["Z" + str(i + 1)])
        elif activations[i] == "swish":
            dZ = dA * swish_back(cache["Z" + str(i + 1)])
        elif activations[i] == "sigmoid":
            dZ = dA * sigmoid_back(cache["Z" + str(i + 1)])
        elif activations[i] == "tanh":
            dZ = dA * tanh_back(cache["Z" + str(i + 1)])
        elif activations[i] == "softmax":
            dZ = cache["A" + str(L)] - Y
        # ==========dW===========
        dW = (1 / m) * (np.dot(dZ, cache["A" + str(i)].T) + lambd * W["W" + str(i + 1)])
        # ==========db===========
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # ==========dA===========
        dA = np.dot(W["W" + str(i + 1)].T, dZ)
        # ======gradients========
        grads["dW" + str(i + 1)] = dW
        grads["db" + str(i + 1)] = db
    return grads

# sigmoid derivative
def sigmoid_back(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

# swish derivative
def swish_back(Z):
    return swish(Z) + sigmoid(Z) * (1 - swish(Z))

# tanh derivative
def tanh_back(Z):
    return 1 - np.power(np.tanh(Z),2)

# relu derivative
def relu_back(Z):
    return (Z > 0) * 1

# ===========================parameter update functions=================================
# normal parameter update
def update_params(params, grads, learning_rate):
    W = params["weights"]
    b = params["biases"]
    L = len(W)

    for i in range(L):
        W["W" + str(i + 1)] = W["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        b["b" + str(i + 1)] = b["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
    return {"weights": W, "biases": b}

# parameter updates with Adam
def update_parameters_Adam(params, grads, Mt, Vt, t, learning_rate=0.01,
                           beta1=0.9, beta2=0.999, epsilon=1e-8):
    W = params["weights"]
    b = params["biases"]
    L = len(W)

    # learning rate for this iteration of Adam
    learning_rate_t = learning_rate * (np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))

    for i in range(L):
        # momentum
        Mt["dW" + str(i + 1)] = beta1 * Mt["dW" + str(i + 1)] + (1 - beta1) * grads["dW" + str(i + 1)]
        Mt["db" + str(i + 1)] = beta1 * Mt["db" + str(i + 1)] + (1 - beta1) * grads["db" + str(i + 1)]

        #RMS
        Vt["dW" + str(i + 1)] = beta2 * Vt["dW" + str(i + 1)] + (1 - beta2) * np.square(grads["dW" + str(i + 1)])
        Vt["db" + str(i + 1)] = beta2 * Vt["db" + str(i + 1)] + (1 - beta2) * np.square(grads["db" + str(i + 1)])

        W["W" + str(i + 1)] = W["W" + str(i + 1)] - \
                              learning_rate_t * (Mt["dW" + str(i + 1)] / (np.sqrt(Vt["dW" + str(i + 1)]) + epsilon))

        b["b" + str(i + 1)] = b["b" + str(i + 1)] - \
                              learning_rate_t * (Mt["db" + str(i + 1)] / (np.sqrt(Vt["db" + str(i + 1)]) + epsilon))

    return {"weights": W, "biases": b},Mt,Vt

# ===========================data manipulation functions=================================
# prepares the data for the network
def processData():
    x = pd.read_csv("DataSets/ghouls-goblins-and-ghosts-boo/train.csv")

    typeDic = {"Ghoul":0,"Goblin":1,"Ghost":2}
    inv_typeDic = {v: k for k, v in typeDic.items()}


    x["type"].replace(typeDic, inplace=True)

    y = x["type"]
    del x["type"]
    del x["id"]

    color = x["color"].unique()
    colorDic = {}
    c = 0
    for i in color:
        colorDic[i] = c
        c += 1
    inv_colorDic = {v: k for k, v in colorDic.items()}

    x["color"].replace(colorDic, inplace=True)

    X_train = np.array(x).T
    Y_train = np.zeros((len(X_train[0]), len(typeDic)))

    c = 0
    for i in Y_train:
        i[y[c]] = 1
        c += 1

    Y_train = Y_train.T
    x = pd.read_csv("DataSets/ghouls-goblins-and-ghosts-boo/test.csv")
    x["color"].replace(colorDic, inplace=True)

    X_test = np.array(x).T

    return X_train,Y_train,X_test,inv_typeDic

# divides the dataset into random mini batches according to the given seed
def random_mini_batches(X, Y, batch_size, seed=1):
    np.random.seed(seed)
    mini_batches = []
    m = X.shape[1]

    # randomize the set according to the seed
    permutation = list(np.random.permutation(m))
    X = X[:, permutation]
    Y = Y[:, permutation].reshape((3, m))

    # number of mini batches (-1 if there is a remainder)
    mini_batch_num = m // batch_size

    # divide the set into mini batches
    for i in range(mini_batch_num):
        MX = X[:, i * batch_size: (i + 1) * batch_size]
        MY = Y[:, i * batch_size: (i + 1) * batch_size]
        mini_batches.append((MX, MY))

    # last mini batch
    if m % batch_size != 0:
        MX = X[:, mini_batch_num * batch_size:]
        MY = Y[:, mini_batch_num * batch_size:]
        mini_batches.append((MX, MY))

    return mini_batches

# ===========================validation functions=================================

# print the accuracy of predictions on a given set
def predict(X, Y, params, activations):

    # predict the output
    AL, cache = forward_prop(X, params, activations)

    # compare the predictions to the expected outputs
    P = np.zeros_like(AL)
    P[np.argmax(AL, axis=0),np.arange(len(AL[0]))] = 1

    print("Accuracy: %f %%" % ((np.sum(P*Y) / Y.shape[1]) * 100))

def predict_testSet(X, params, activations,labels):

    # predict the output
    AL,_ = forward_prop(X[1:], params, activations)

    # one-hot encode the predictions
    classes = np.argmax(AL, axis=0)

    # get labels
    predictions = [labels[x] for x in classes]

    #save the predictions
    outFile = pd.DataFrame({'id':X[0].astype(int),'type':predictions})
    outFile.to_csv("output.csv",index=False)
    print("done")



# =================================MAIN====================================

if __name__ == '__main__':
    X_train,Y_train,X_test,labels = processData()

    # number of layers (other than the input), and the number of hidden units
    layer_dims = [18, 32, 32, 42, 18, 3]

    # the activation function of each layer
    activations = ["swish", "tanh", "swish", "swish", "relu", "softmax"]

    # train the network and save the parameters

    # train(X_train, Y_train, layer_dims, activations,learning_rate=0.005,
    #       mini_batch_size=32, epochs=350,lr_decay=0.003273,
    #       optimizer="adam",report=True,save=True)

    # load the learned parameters
    params = pickle.load(open('learned_Parameters',"rb"))

    # print accuracy (for debugging)
    predict(X_train, Y_train, params, activations)

    # prepare submission
    predict_testSet(X_test,params,activations,labels)