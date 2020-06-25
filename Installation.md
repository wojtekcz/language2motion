# local setup

## 1. setup container
+ install docker

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

TODO: add tensorboard requirements

```
./docker_build.sh download
```

TODO: move following section out
or: download toolchain, and use for building docker image
TODO: set local ip for serving toolchain tgz

```
./docker_build.sh cached
```


### c. create (start) docker container

TODO: check docker_run.sh on linux
```
./docker_run.sh cpu_macos
```

start

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
tar xzvf motion_dataset_v3.norm.10Hz.tgz

wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt
```

### d. run Motion2label script

```
cd ../code
swift run Motion2label
```

### or use jupyter lab

open link in google chrome

## 3. vscode integration

### a. install vscode, do clean install, how?

### b. open vscode

### c. install extensions
- Remote-Container
- Remote-SSH

### d. attach to running container
TODO: screenshot



### e. install extensions (in container)
- CodeLLDB
- Maintained Swift Development Environment


### f. open folder "/" in vscode


### FIXME: how to use "remote settings location"

### g. open workspace

TODO: kill "_wrk"
/notebooks/language2motion.gt/code/l2m_wrk.code-workspace

TODO: screenshot

run/debug Motion2label

## 4. (optional) start tensorbord

exec into container

cd tboard

start tensorboard

use tensorboard
TODO: screenshot
