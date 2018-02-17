# Dinosaurus Island - Character Level Language Model

## Gradient Clipping

* LSTMs handle vanishing gradient problems but not the Exploding Gradient Problems
* Gradient Clipping is one way to solve Exploding Gradient Problem

```python
def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    '''
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)

    clipped_gradients = {}
    for g in ['dWax', 'dWaa', 'dWya', 'db', 'dby']:
        clipped_gradients[g] = np.clip(gradients[g], -maxValue, maxValue)

    return clipped_gradients
```

## Sampling

* Numpy provides `np.random.choice` to perform sampling based on a probability distribution
```python
np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
index = np.random.choice([0, 1, 2, 3], p = p.ravel())
```

## Optimization

```python
# Forward propagate through time (≈1 line)
loss, cache = rnn_forward(X, Y, a_prev, parameters)

# Backpropagate through time (≈1 line)
gradients, a = rnn_backward(X, Y, parameters, cache)

# Clip your gradients between -5 (min) and 5 (max) (≈1 line)
gradients = clip(gradients, 5)

# Update parameters (≈1 line)
parameters = update_parameters(parameters, gradients, learning_rate)
```

## Resources

* [Andrej Karpathy Implementation](https://gist.github.com/karpathy/d4dee566867f8291f086)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Keras LSTM Text Generation](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)
