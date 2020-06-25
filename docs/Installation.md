# local setup

## 1. setup container
### 0. install/setup docker
+ dataset in memory requires about 6GB RAM

assuming ~/swift4tf as root for installations

### a. clone swift-jupyter repo
```
mkdir ~/swift4tf
cd ~/swift4tf
git clone https://github.com/wojtekcz/swift-jupyter.git swift-jupyter.gt
cd swift-jupyter.gt
git checkout --track origin/language2motion

```
### b. build docker image
```
./docker_build.sh
```

### c. create (start) docker container

TODO: check docker_run.sh on linux

```
./docker_run.sh macos|gpu
```

### c. start
```
docker start swift-jupyter
```

## 2. setup project sources and data

### a. exec into container
```
docker exec -it swift-jupyter bash 
```

### b. clone language2motion repo
```
git clone https://github.com/wojtekcz/language2motion.git language2motion.gt
```

### c. download data for Motion2label and Language2label scripts
```
cd /notebooks/language2motion.gt/data/
wget https://github.com/wojtekcz/language2motion/releases/download/v0.2.0/motion_dataset_v3.norm.10Hz.tgz
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt
tar xzvf motion_dataset_v3.norm.10Hz.tgz
```

### d. run Motion2label script
```
cd ../code
swift run Motion2label
```

### e. or use jupyter lab

open link in google chrome

## 3. vscode integration

### a. install vscode, do clean install, how?

### b. open vscode

### c. install extensions
- Remote-Container

TODO: - Remote-SSH, for colab integration

### d. attach to running container
TODO: screenshot

### e. install extensions (in container)
- CodeLLDB
- Maintained Swift Development Environment

TODO: screenshot

reload

### f. open folder ```/``` in vscode

### FIXME: how to use "remote settings location"

### g. open workspace
```
/notebooks/language2motion.gt/code/l2m.code-workspace
```

TODO: screenshot

### h. run/debug Motion2label

TODO: screenshot

## 4. (optional) tensorbord

### a. exec into container

### b. start tensorboard
```
cd /notebooks/language2motion.gt/data/
tensorboard --bind_all --logdir tboard/
```

### c. use tensorboard

TODO: screenshot
