# Lecture 4: Backpropagation

Modern deep learning frameworks compute gradients for you. However, backpropagation doesn't always work perfectly, and understanding why is crucial for debugging and improving models.

## Things to Know

### Regularization

The use of regularization (largely) prevents overfitting when we have a lot of features, or develop a very powerful/deep model. When you have a model with a lot of parameters, the model gets very good at predicting the answers based on the data with which you trained it. However, in practice, the model can become very poor at working in the real world on different examples and sets of data. Using large numbers of parameters, even on small quantities of training data/sets, only works if you regularize the model (and it works very well).

### Vectorization

Always try to use vectors and matrices rather than for loops.

## Additional Resources

Medium article covering more on backpropagation, and why you should understand it: 

[Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)