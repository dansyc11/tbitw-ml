# TBITW ML Library

*A lightweight machine learning library built from scratch in C using only the standard library*[^1][^2]

<p align="center">
  <img src="docs/mnist_demo.gif" alt="MNIST Demo" width="600"/>
</p>

## Results

| Dataset | Accuracy | 
|---------|----------|
| **MNIST Digits** | 92.37% | 
| **Fashion-MNIST** | 92.16% | 

I have provided 2 pre-trained models so you can instantly demo it without waiting to train!

## Quick Start

```bash
git clone https://github.com/dansyc11/tbitw-ml
cd tbitw-ml
make

# Try the pre-trained MNIST model
./mnist_predict
# Enter model: # mnist_adam_9257.tbitw or fashion_adam_9216.tbitw
# Enter index: # any from 0 to 9999
```

## Live Demo

### Digit MNIST Classification

<p align="center">
  <img src="docs/digit_example.png" alt="MNIST Prediction Example" width="700"/>
</p>

The classifier visualizes digits in ASCII and shows confidence scores in real-time.

### Fashion MNIST Classification

<p align="center">
  <img src="docs/fashion_prediction.png" alt="Fashion-MNIST Example" width="700"/>
</p>

```bash
# Classify clothing items
./fashion_predict
# Enter model: fashion_adam_9216.tbitw
# Enter index: 100
```


## Technical Details

See [Technical Documentation (PDF)](docs/technical_reference.pdf) for architecture details and how the algorithms were implemented.


---

[^1]: The name TBITW stands for The Best Thing In The World suggested by a friend during a random discord call

[^2]: Built with inspiration from Tsoding and Magicalbat's ML tutorials. Core machine learning algorithm implementations written from scratch with educational guidance from their teaching approach.
