# Language2motion

The goal of this project is to create multi-modal implementation of Transformer architecture in Swift. It's a learning exercise for me, so I'm taking it slowly, starting from simple image classifier and building it up.

Also it's an attempt to answer the question if Swift for Tensorflow is ready for non-trivial work.

The use-case is based on a paper "[Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks"](https://arxiv.org/abs/1705.06400) by Matthias Plappert. He created a nice dataset of few thousand motions "[The KIT Motion-Language Dataset (paper)](https://arxiv.org/abs/1607.03827)", [website](https://motion-annotation.humanoids.kit.edu/dataset/).

## The rough plan
- something 2 label
  - image 2 label
    - [x] build image2label dataset with images representing motions
    - [x] assign 5 dummy(ish) classes with PCA and k-means on motion annotations
    - [x] classify motion images (+in fastai, +in swift)
  - language 2 label
    - [x] Transformer encoder on annotation + classifier
    - [x] batched prediction
    - [x] Use BERT classifier to assign better labels - didn't work
    - [x] manually assign better labels
  - motion 2 label
    - [x] 1-channel ResNet on motion + classifier
    - [x] ResNet feature extractor + Transformer encoder on motion features + classifier - didn't work
    - [x] Transformer encoder on motion + classifier
- language 2 language
    - [x] Transformer seq2seq from annotation to label text
    - [x] Transformer seq2seq from annotation to (same) annotation
- motion 2 language
  - [x] Transformer from motion to annotation
  - [ ] Transformer from motion to label text
- language 2 motion
  - Transformer encoder on annotation
  - Transformer decoder on motion

## Dataset files
* original: [2017-06-22.zip](https://motion-annotation.humanoids.kit.edu/downloads/4/)
* processed: 
  * [motion_dataset_v3.norm.10Hz.tgz](https://github.com/wojtekcz/language2motion/releases/download/v0.2.0/motion_dataset_v3.norm.10Hz.tgz)

* annotations and labels: 
  * [labels_ds_v2.csv](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv)
  * [labels_ds_v2.balanced.515.csv](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.balanced.515.csv)
  * [vocab.txt](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt)


## Motion player
* [Mokka C3D file player](http://biomechanical-toolkit.github.io/mokka/index.html)

## Runtime env
* [custom swift-jupyter](https://github.com/wojtekcz/swift-jupyter/tree/language2motion)
