import numpy as np
import logging

class NNLayer:
    def __init__(self,inputs,outputs):
        #Initialize the weights based on the inputs and outputs dimensions
        self.input_dim = inputs
        self.outputs_dim = outputs
        self.weights = np.random.rand(inputs,outputs)
        
    def forward(self,x):
        self.input = x
        if x.shape[0] != self.input_dim:
            logging.warn("Dimensional mismatch")
            return
        self.output = self._sigmoid(np.dot(x,self.weights))
        return self.output
    
    def _sigmoid(self,x):
        '''Create sigmoid activation function'''
        return 1/(1+np.exp(-x))
    
    def _sigmoid_derivative(self,x):
        '''Derivative of sigmoid activation'''
        z = self._sigmoid(x)
        return z*(1-z)
    
    def _transpose_vector(self,x):
        if len(x.shape) == 1:
            input = x.reshape(-1,1)
        else:
            input = x.T
        return input
            
    def back_prop(self,back_x):
        '''Propogate the loss backwards using chain rule and return variable.'''
        self.back_x = self._sigmoid_derivative(self.output)*back_x        
        self.d_weights = np.dot(self.input.reshape(-1,1),self.back_x.reshape(-1,1).T)
        self.back_x = np.dot(self.d_weights.reshape(-1,1),self.back_x.reshape(-1,1).T).T[0]
        return self.back_x
    
class createModel:
    def __init__(self,layers):
        '''Given a list of layers define the model [2,4,2] in our case'''
        self.model = self.build_model(layers)
        
    def build_model(self,layers):
        '''Given a list of layers define the model [2,4,2] in our case'''
        model = np.array([])
        for i in range(len(layers)-1):
            l = NNLayer(layers[i],layers[i+1])
            model = np.append(model,l)
        return model
    
    def forward(self,x):
        '''Forward pass through all the layers.'''
        for layer in self.model:
            x = layer.forward(x)
        self.output = x
        return self.output

    def backward(self):
        try:
            back_x = self.loss_backward
            for layer in reversed(self.model):
                back_x = layer.back_prop(back_x)
                layer.weights += layer.d_weights
        except: 
            logging.warning("loss may not yet be defined.") 
            return
        
    def _sigmoid(self,x):
        '''Create sigmoid activation function'''
        return 1/(1+np.exp(-x))
    
    def _sigmoid_derivative(self,x):
        '''Derivative of sigmoid activation'''
        z = self._sigmoid(x)
        return z*(1-z)
    
    def compute_loss(self,y,output):
        '''Forward pass through all the layers.'''
        self.loss = ((y-output)**2).sum()
        #needed for backprop
        self.loss_backward = 2*(y-output)
        return self.loss

def detailed_train_loop():
    #define model
    layer1 = NNLayer(2,4)
    layer2 = NNLayer(4,1)

    #define input data
    input = np.array([1,1])
    target = 1

    for i in range(10):
        #propogate forward
        x = layer1.forward(input)
        x = layer2.forward(x)
        
        #compute the loss
        loss = (target-x)**2
        print("output:",x,"target:",target,"loss:",loss)
        
        #gradient decent and back prop
        back_x =  2*(target-x)
        back_x = layer2.back_prop(back_x)
        back_x = layer1.back_prop(back_x)

        #update weights
        layer2.weights += layer2.d_weights
        layer1.weights += layer1.d_weights

def plot_graph(data):
    import matplotlib.pyplot as plt 
    plt.plot(data) # plotting by columns
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()
    
if __name__ == '__main__':
    #define input data
    input = np.array([1,1])
    target = 0.5
    loss_ = []
    
    model = createModel([2,30,100,30,10,1])
    
    for i in range(10):
        output = model.forward(input)
        loss = model.compute_loss(target,output)
        model.backward()
        loss_.append(loss)
        print("output:",output,"loss:",loss)
    
    # plot_graph(loss_)