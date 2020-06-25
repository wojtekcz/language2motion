# TODO: paste portmap.io instructions

open notebook swift ssh on colab, 
make sure runtime type is gpu
run notebook

scp rsa key into colabp
scp ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local colabp:/root/.ssh

ssh into colabp
```
ssh colabp
```

# TODO: paste ~/.ssh/config content

```
tmux -CC
```

chmod 600 ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local

in vscode
connect to ssh host

in vscode install:
- codelldb extension
- Maintained Swift Development Environment

open folder "/" in vscode

ssh-agent bash
ssh-add /root/.ssh/id_rsa_wcz\@MacBook-Wojtka.local

cd /content
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"
git clone git@github.com:wojtekcz/language2motion.git language2motion.gt

open workspace in vscode

// to build project in terminal
cd language2motion.gt/code/
/swift/toolchain/usr/bin/swift build -c release

// get dataset
cd /content/language2motion.gt/data/
wget https://github.com/wojtekcz/language2motion/releases/download/v0.2.0/motion_dataset_v3.norm.10Hz.tgz
tar xzvf motion_dataset_v3.norm.10Hz.tgz

wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/labels_ds_v2.csv
wget https://github.com/wojtekcz/language2motion/releases/download/v0.1.0/vocab.txt

// start tensorboard
pip install google-auth-oauthlib==0.4.1
pip install grpcio==1.24.3
cd /content/language2motion.gt/data
tensorboard --logdir tboard
