import numpy as np
from tqdm import trange

def mini_batches(inputs, targets, batch_size, shuffle=False):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def binary_cross_entropy(yhat,y):
    return -np.mean(y*np.log(yhat) + (1 - y)*np.log(1-yhat))


class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.training_log = []
        self.validation_log = []
    
    def add(self, layer):
        self.layers.append(layer)

    def predict(self,layer,X):
        activations = []
        input = X
        for i in range(len(layer)):
            activations.append(layer[i].forward(X))
            X = layer[i].forward(X)
        yhat = activations[-1]
        return np.heaviside(yhat-0.5, 0)

    def fit(self, x_train, y_train, x_val=None, y_val=None , epochs=10):

        for epoch in range(epochs):
            for x_batch,y_batch in mini_batches(x_train,y_train,batch_size=32,shuffle=True):
                #train(network,x_batch,y_batch)
                # layer_activations = forward(self.layers,x_batch)
                activations = []
                for i in range(len(self.layers)):
                    activations.append(self.layers[i].forward(x_batch))
                    x_batch = self.layers[i].forward(x_batch)
                yhat = activations[-1]

                loss = binary_cross_entropy(yhat,y_batch)
                loss_grad = yhat - y_batch

                for i in range(1,len(self.layers)):
                    loss_grad = self.layers[len(self.layers) - i].backward(activations[len(self.layers) - i - 1], loss_grad)
            self.training_log.append(np.mean(self.predict(self.layers,x_train) == y_train)) 
            self.validation_log.append(np.mean(self.predict(self.layers,x_val) == y_val))
            print("Epoch : ",epoch)
            print("Training Accuracy: ", self.training_log[-1])
            print("Validation Accuracy: ", self.validation_log[-1])
        
        return (self.training_log,self.validation_log)




