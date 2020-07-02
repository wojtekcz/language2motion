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

### 0. setup ssh connection ```~/.ssh/config```
```
# ============= colab via portmap.io ===============
# ssh -i ~/.ssh/wojtekcz.first.pem wojtekcz.first@wojtekcz-22423.portmap.io -N -R 22423:localhost:22
Host colabp
HostName wojtekcz-22423.portmap.io
User root
Port 22423
StrictHostKeyChecking no
UserKnownHostsFile /dev/null
IdentityFile ~/.ssh/private_key.pem
LocalForward 6006 localhost:6006
```

### a. ssh into colabp
```
ssh colabp
```

### b. (optional) run tmux in iterm
```
tmux -CC
```

### c. (optional) load github credentials
```
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"
chmod 600 ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
ssh-agent bash
ssh-add /root/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
```
<!-- ## 3. setup project sources and data
### b. clone language2motion repo
```
# cd /content
git clone https://github.com/wojtekcz/language2motion.git language2motion.gt
``` -->

<!-- ### c. download data for Motion2label and Language2label scripts
```
cd /content/language2motion.gt/data/
wget https://github.com/wojtekcz/language2motion/releases/download/v0.2.0/motion_dataset_v3.norm.10Hz.tgz
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.balanced.515.csv
tar xzvf motion_dataset_v3.norm.10Hz.tgz
``` -->

### d. run Motion2label script
```
cd /content/language2motion.gt/code
swift run -c release Motion2label
swift run -c release Lang2lang
```

## 4. vscode integration

### 0. install extension
- Remote-SSH

### a. Remote-SSH: Connect to Host... ```colabp```

### b. install extensions (in container)
- CodeLLDB
- Maintained Swift Development Environment

<!-- ### c. open folder ```/``` -->

### c. File/Open workspace...
```
/content/language2motion.gt/code/l2m.code-workspace
```

## 5. (optional) tensorbord

### a. start tensorboard
```
cd /content/language2motion.gt/data
tensorboard --logdir tboard
```
