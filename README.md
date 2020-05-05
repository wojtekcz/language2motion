# Language2motion

The goal of this project is to create multi-modal implementation of Transformer architecture in Swift. It's a learning exercise for me, so I'm taking it slowly, starting from simple image classifier and building it up.

Also it's an attempt to answer the question if Swift for Tensorflow is ready for non-trivial work.

The use-case is based on a paper "[Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks"](https://arxiv.org/abs/1705.06400) by Matthias Plappert. He created a nice dataset of few thousand motions "[The KIT Motion-Language Dataset](https://arxiv.org/abs/1607.03827)".

## The rough plan
1. \+ build image2label dataset with images representing motions
2. \+ assign 5 dummy(ish) classes with PCA and k-means on motion annotations
3. \+ classify motion images (+in fastai, +in swift)
4. \- do image captioning (with RNN decoder)
5. RNN encoder on motion + classifier
6. Transformer encoder + classifier
7. Transformer encoder on motion
8. Transformer decoder on motion

## Processed dataset
* [img2label_ds_v1.tgz](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/img2label_ds_v1.tgz)
