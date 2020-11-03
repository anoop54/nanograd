# nanograd
A insanely simple and small NN engine for forward and backward propagation, using nothing but Numpy. Developed for the teaching McMaster University students the fundamentals of machine learning.

## API 
model = createModel([2,30,100,30,10,1]) -> defines the model  
output = model.forward() -> forward propagation  
loss = model.compute_loss(target,output) -> compute final loss  
model.backward() -> update weights based on back prop  

So we have  
-model.forward()  
-model.compute_loss()  
-model.backward()  
