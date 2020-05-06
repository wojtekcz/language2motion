# Language2motion

The goal of this project is to create multi-modal implementation of Transformer architecture in Swift. It's a learning exercise for me, so I'm taking it slowly, starting from simple image classifier and building it up.

Also it's an attempt to answer the question if Swift for Tensorflow is ready for non-trivial work.

The use-case is based on a paper "[Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks"](https://arxiv.org/abs/1705.06400) by Matthias Plappert. He created a nice dataset of few thousand motions "[The KIT Motion-Language Dataset (paper)](https://arxiv.org/abs/1607.03827)", [website](https://motion-annotation.humanoids.kit.edu/dataset/).

## The rough plan
- motion 2 language
  - image 2 label
    - [x] build image2label dataset with images representing motions
    - [x] assign 5 dummy(ish) classes with PCA and k-means on motion annotations
    - [x] classify motion images (+in fastai, +in swift)
  - motion 2 label
    - [ ] \* Transformer encoder on motion + classifier
    - [ ] RNN encoder on motion + classifier
  - motion 2 language
    - do image captioning (with RNN decoder)
- language 2 motion
  - Transformer encoder on annotation
  - Transformer decoder on motion

## Dataset files
* original: [2017-06-22.zip](https://motion-annotation.humanoids.kit.edu/downloads/4/)
* processed: [img2label_ds_v1.tgz](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/img2label_ds_v1.tgz)
