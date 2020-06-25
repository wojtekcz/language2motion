# How to use colab from local vscode?
TODO: paste portmap.io instructions

## 1. setup colab VM

### a. notebook swift ssh

* open notebook swift ssh on colab, 
* make sure runtime type is gpu
* run notebook up to ssh tunnel cell

### b. scp rsa key into colabp
```
scp ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local colabp:/root/.ssh
```

## 2. ssh/terminal

### 0. setup ssh connection
TODO: paste ~/.ssh/config content

### a. ssh into colabp
```
ssh colabp
```

### b. run tmux in iterm
```
tmux -CC
```

### c. (optional) load github credentials
```
chmod 600 ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
ssh-agent bash
ssh-add /root/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
```
## setup project sources and data
### b. clone language2motion repo
```
cd /content
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"
git clone git@github.com:wojtekcz/language2motion.git language2motion.gt
```

### c. download data for Motion2label and Language2label scripts
```
cd /content/language2motion.gt/data/
wget https://github.com/wojtekcz/language2motion/releases/download/v0.2.0/motion_dataset_v3.norm.10Hz.tgz
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt
tar xzvf motion_dataset_v3.norm.10Hz.tgz
```

### d. run Motion2label script
```
cd ../code
swift run -c release Motion2label
```

## 3. vscode integration

### a. install extension
- Remote-Container

### b. connect to ssh ```colabp``` host

### c. install extensions (in container)
- CodeLLDB
- Maintained Swift Development Environment

### d. open folder ```/```

### e. open workspace
```
/content/language2motion.gt/code/l2m.code-workspace
```

## 4. (optional) tensorbord


### a. instal deps
```
pip install google-auth-oauthlib==0.4.1 grpcio==1.24.3
```

### b. start tensorboard
```
cd /content/language2motion.gt/data
tensorboard --logdir tboard
```
