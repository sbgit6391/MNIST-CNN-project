Three iterations of handwritten digit classification using the MNIST data set.
One where the back propagation algorithm was derived and written from scratch using Numpy. 
A second, still written in Numpy, but given many feature and network architecture improvements. 
And a third which implemented a Convolutional Neural Network using PyTorch. 

1. First Network: Architecture

I used a shallow neural network of dimensions 784-30-10 (single hidden layer). The loss function used was quadratic cost,
and the weights and biases were initialised completely randomly to a gaussian distribution. 

2. First Network: Derivation of back propagation algorithm

A handwritten explanation of the derivations of back propagation and the motivations behind its origins is attached. 
It also goes on to describe some of the mathematics of the changes I went on to make from the first to second network. 

3. Second Network: The changes I made and why

I changed the quadratic cost model to a cross-entropy cost model or a log-likelihood model with a softmax layer 
(these features are mathematically defined and justified in the handwritten document). I made these changes to address
learning slowdowns that arose from initial neuron saturation. I initialised the weights with a scaling 
factor of the reciprocal of the square root of the number of weights in the network to prevent exploding gradients caused by 
large initial weights. I also implemented L2 regularisation to prevent over-fitting to the training data, and finally I implemented 
a momentum based stochastic gradient descent algorithm to damp the effects of oscillations in Cost Function valleys. 

4. Second Network: Results.

With the updated features the network was able to train relatively quickly and obtain a classification accuracy of ~94% on the test data.
I was pleased with this as the hyperparameters had been chosen very roughly. 

5. Final Network: Architecture.

The final network had a fundamentally different structure to that of the first, instead of fully connected but shallow neural networks
I began to investigate whether deeper neural networks with multiple hidden layers could learn effectively. Through experimentation and 
display of the gradients in different layers it became clear that as the signal was back propagated through the network its attenuation
was significant such that the earliest layers of the network learnt very slowly in comparison to terminal layers. 

The solution was to consider a different type of deep learning: the convolutional neural network. It doesn't discard the geographical 
location of pixels as earlier models did, instead it takes small samples of the larger grid and forms many feature maps.  
Each small sample of the larger grid has the same weight and bias templates such that each feature map learns one individual feature. 
The information stored in these maps is then distilled and used for deep learning. 

6. Final Network: Results

Implementing this network in PyTorch allowed me to reach a classification accuracy of ~99% within 5 epochs of training, 
a very impressive result. 

7. Three things I would do differently next time

I would implement a bayesian algorithm/ protocol for hyper-parameter designation as network architecture in terms of number of neurons as
well as learning rates, regularisation constants and momentum parameters were all chosen to be sensible but not fine tuned- accuracy and
speed can be found by tuning. I would choose a harder data set than MNIST, its a classic problem so by choosing a more challenging less 
perfectly arranged data set it can test my abilities on real-world problems. Finally, I might attempt to implement a variable learning rate 
and momentum parameter to vary as the back-propagation approaches cost minima to avoid overshooting. 
