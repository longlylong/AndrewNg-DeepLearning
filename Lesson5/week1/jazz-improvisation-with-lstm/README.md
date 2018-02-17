# Improvise a Jazz Solo with an LSTM Network

## Problem Statement

* Note Representation: pitch AND duration
* In practice, it is more sophisticated: one might press down two piano keys at once
* The model is trained on random snippets of 30 values taken from a much longer piece of music

## Model

* `Lambda` can be used to define arbitrary operation in Keras
```python
x = Lambda(lambda X: X[:,t,:])(X)  # select the "t"th time step vector from X
```
* Reshaping a tensor with `Reshape`
```python
reshapor = Reshape((1, 78))
```
* Model creation with `Model`
```python
model = Model(inputs=[X, a0, c0], outputs=outputs)  # Multiple inputs are allowed
model.fit([X, a0, c0], list(Y), epochs=100)
```

## Sampling new Music

* Use trained model to get output at time = t
* Sample from generated probabilities
* Feed to next time t + 1
* Initialize `a`, `x` and `c` with zeros
* Post processing is needed to transform one hot encoded note into actual pitch and duration

## Resources

* Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
* Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
* Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
* François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)
