# Language2motion

The goal of this project is to create multi-modal implementation of Transformer architecture in Swift. It's a learning exercise for me, so I've taken it slowly, starting from simple image classifier and building it up.

Also it's an attempt to answer the question if Swift for Tensorflow is ready for non-trivial work.

The use-case is based on a paper "[Learning a bidirectional mapping between human whole-body motion and natural language using deep recurrent neural networks"](https://arxiv.org/abs/1705.06400) by Matthias Plappert. He created a nice dataset of few thousand motions "[The KIT Motion-Language Dataset (paper)](https://arxiv.org/abs/1607.03827)", [website](https://motion-annotation.humanoids.kit.edu/dataset/).

The Motion2Language Transformer which works is there, already. Lang2motion one started to work recently, and I'm implementing more sophisticated motion generation strategy now.

I'm using modified [Swift Transformer implementation](https://github.com/acarrera94/swift-models/tree/translation/Models/Translation) by Andre Carrera.

## The plan
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
- language 2 motion
  - [x] Transformer encoder on annotation
  - [x] Transformer decoder on motion

## Dataset files
* original: [2017-06-22.zip](https://motion-annotation.humanoids.kit.edu/downloads/4/)
* processed: 
  * [motion_dataset_v3.10Hz.tgz](https://github.com/wojtekcz/language2motion/releases/download/v0.3.0/motion_dataset_v3.10Hz.tgz)
  * [motion_dataset_v3.10Hz.midi.tgz](https://github.com/wojtekcz/language2motion/releases/download/v0.3.0/motion_dataset_v3.10Hz.midi.tgz)
  * [motion_dataset_v3.10Hz.mini.tgz](https://github.com/wojtekcz/language2motion/releases/download/v0.3.0/motion_dataset_v3.10Hz.mini.tgz)

* annotations and labels:
  * [labels_ds_v2.csv](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv)
  * [labels_ds_v2.balanced.515.csv](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.balanced.515.csv)
* vocabulary
  * [vocab.txt](https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt)


## Motion player
* [Mokka C3D file player](http://biomechanical-toolkit.github.io/mokka/index.html)

## Runtime env
* [custom swift-jupyter](https://github.com/wojtekcz/swift-jupyter/tree/language2motion)

* [Installation](docs/Installation.md)

* [connect_colab](docs/connect_colab.md)
* [connect_gcp_instance](docs/connect_gcp_instance.md)
