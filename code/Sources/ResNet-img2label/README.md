# ResNet-50 with img2label

This example demonstrates how to train the [ResNet-50 network]( https://arxiv.org/abs/1512.03385) against the [img2label dataset](https://github.com/wojtekcz/language2motion/releases/tag/v0.1.0).

A modified ResNet-50 network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the CIFAR-10 dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To train the model, run:

```sh
cd swift-models
swift run -c release ResNet-img2label
```
