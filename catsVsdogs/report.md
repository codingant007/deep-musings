# CNN for classifying cat and dog images

##First attempt:

CNN architechture:
- 2 conv pool layers
	- 50 (5x5) kernels, 2x2 maxpooling, relu
	- 100 (5x5) kernels, 2x2 maxpooling, relu
- 2 fully connected layers
	- 500 nodes, relu
	- 10 nodes, relu
- softmax

validation set:

- 0.2 of training data

Minibatch size:

- 128

Learning rate:

- 0.1

Cost function:

- negative log likelihood

Optimization:

- minibatch SGD

##Training problem:
While training minibatch cost would increase exponentially starting with 2.3 to 5347e^2134 after 10 minibatch iterations then nan.

###Reasoning:
Figured this could be due high learning rate.

###Solution:
Experimented by decreasing learning rate to 0.01 and so on to 0.001.

###Revision:
set learning rate to 0.001

###Result:
The cost of minibatch seemed unstable in the beginning with values as large as 10k but soon stabilized to 0.6 after around 1000 iterations. Also observed that validation loss was decreasing steadily.

##Overfitting:
After around 5000 iterations cost of minibatch started decreasing with a greater pace than earlier and validation loss began to increase. After 500 more iterations was decreased to a very small value in the order of e^-12 and validation loss did not decrease. Then figured that the model was overfitting to train data.

###Revision:
Introduced drop out in the layers. Tried values in the range of 0.5 to 0.8.
Also experimented with L1 and L2 norm without dropout but thier performance wasnt as remarkable as dropout.

###Result
Overfitting problem was overcome and model could be trained smoothly.

##Paralellism:
The GPU utilizations was 75%. So, there was more scope for parallelism.

###Revision:
Increase batchsize to 256.

###Result:
Training pace increased. This could be due to increased parallellism and more training examples participating in a single iteration leading to movement in a direction more inclined toward the solution.

##Low converge rate:
Upon reaching validation accuracy of 75% at around 40k iterations, I felt the convergence is present but at a very low pace I could also see the learning rate being printed after every epoch which is quite low. So I figured, Introducing momentum and veocity could be a good idea.

###Revision:
velocity and momentum are introduced into learning process 

Velocity is used to update parameters in the following way

```
param = param - velocity
```

Velocity is updated after each iteration in the following way
```
velocity = momentum * velocity + learning_rate * grad
```
Momentum is also updated periodically after each epoch instead of iteration
```
momentum = momentum + (momentum_limit - momentum)/32
```

###Result:
As expected, convergence rate increased.


##Alternative Cost function:
Tried hinge loss instead of negative log likelihood loss. 

###Result:
Ran for 10000 iterations with similar performance as the latter so choose not to procceed with the experiment there after.


##Sticking to the simple architechture rather than higher number of layers:
The simple model with 2 convpool layers and 2 fully connected layers has managed to learn the binary classification well enough to have reached a validation accuracy of **85.6%**. This accuracy would further be increased if training is left to proceed as the training was terminated before the model overfit to the train data. So, the model can be further trained. But the catch here is that, the convergence rate is very low. We can conclude two points from the above observations.

1. Just 2 covpool layers along with 2 fully connected layers are good enough to learn the representations involved in cat vs dog classification. Adding layers might not add any power in this scenario.

2. In this context achieving better accuracy is more a matter of converging well. So, applying an upgrade to momemtum upgrade like adadelta or RMSdrop is the way to go.

