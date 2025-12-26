---
title: "Deep Learning Basics - 01"
date: 2025-09-15
tags: [deeplearning, machine-learning]
draft: true
math: true
---

## Perceptron: Output propogation
A Perceptron: It is a single neuron with weighted sum, It computes a weighted sum of inputs, adds a bias, then applies an activation function

$\hat{y} = g\Big(w_0+\sum_{i=1}^{m} x_i w_i\Big)$

$\hat{y} = g\Big(w_0+X^TW\Big)$

y = g(z);\
g(): Activation function to add non linearity to the neuron\
z: The result of dot product plus the bias  with a activation function (nonlinearity)

Take a dot product apply a bias and apply a non linearity and keep repeat over

## Multi output perceptron
To predict multile outputs

$z_i = w_{0,_i} + \sum_{j=1}^m x_i w_j,_i$

## Single layer neural network
$z_i = w_{0,i}^{(1)}+\sum_{j=1}^{m}x_j w_{j,i}^{(1)}$

$\hat{y}_i=g\Big(w_{0,i}^{(2)}+\sum_{i=1}^{d1}g(z_i)+w_{j,i}^{(2)}\Big)$


**Neural Network Loss**: How far the prediction is from the ground truth of the data
The smaller the loss the close the ouput prediction and actual truth\
To train a neural network / model we need to know the bad predictions

**Remeber**: OUR LOSS IS A FUNCTION OF NETWORK WEIGHT

LOSS FUCNTION CAN BE DIFFICULT TO OPTIMIZE

## Qunatifying loss
The loss of network meaures the cose incurred from incorrect prediction and
we need to qunatify how bad predictions is the output vs how good it is(*how close the output prediction and the acutal output*)

$L\Big(f(x^{i};W),y^{(i)}\Big)$

Where
- $f\Big(x^{i};W\Big)= Predicted$
- $y^{i}=Actual$

## Empirical Loss (object function / Cost function / Emperical risk)
Total loss over entire dataset

$J(W) = \frac{1}{n}\sum_{i=1}^{n}L\Big(f(x^{i};W),y^{(i)}\Big)$

Where:
- $y^{(i)}  = Actual$
- $f(x^{(i)};W)  = Predicted$

## Binary cross entropy loss
Can be used with models which output probability b/w 0 and 1

$ J(W) =\frac{-1}{n}\sum_{i=1}^{n}y^{(i)} log\Big(f(x^{i}; W)\Big)+(1-y^{(i)})log\Big(1-f(x^{i};W)\Big)$

Where:
- $y^{(i)}  = Actual$
- $f(x^{(i)};W)  = Predicted$

## Mean Squared Error Loss
It can be used with regression models than prouduces continous real number's

$J(W)=\frac{1}{n}\sum_{i=1}^{n}\Big(y^{(i)}-f(x^{(i)};W)\Big)^2$

Where
- $y^{i} = Actual Truth$
- $f(x^{i}; W) = Predicted$

## Loss Optimization
We want to find the network weights to achieve the lowest loss

$W^{*} = argmin_{m}\frac{1}{n}\sum_{1=1}^nL\Big(f\Big(x^{(i)};W\Big), y^{(i)}\Big)$

$W^{*} = argmin_{m}J(W)$

Where
- $W=\{W^{(0)},W^{(1)},.....\}$

**Compute weight / Adjust weight**:
$\frac{\partial J(W)}{\partial W}.$
### Gradient descent Alogrithm
- Select random weight's ~ $N(0,\sigma^{2})$
- Loop until convergence(meeting point)
- Compute gradient $\frac{\partial J(W)}{\partial W}$
- Update weights $W \leftarrow W - n \frac{\partial J(W)}{\partial W}$ (n = Ada step size)
- Return Weights

### Stochastic Gradient descent
- Select random weight's ~ $N(0,\sigma^{2})$
- Loop until convergence(meeting point)
- Pick data point i
- Compute gradient $\frac{\partial J_{i}(W)}{\partial W}$
- Update weights $W \leftarrow W - n \frac{\partial J(W)}{\partial W}$ (n = Ada step size)
- Return Weights

### Mini batched Gradient descent
- Select random weight's ~ $N(0,\sigma^{2})$
- Loop until convergence(meeting point)
- Pick batch of B data points
- Compute gradient $\frac{\partial J(W)}{\partial W} = \frac{1}{b}\sum_{k=1}^B\frac{\partial J_{k}(W)}{\partial W}$
- Update weights $W \leftarrow W - n \frac{\partial J(W)}{\partial W}$ (n = Ada step size)
- Return Weights

Mini bathces leads to train faster parallelize computation + achieve speed increase on GPU

## Regularization
Helps models in fitting the data and prevent from overfitting

**Drouputs**
-
