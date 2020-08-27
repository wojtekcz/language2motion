# How to use colab from local vscode?
TODO: paste portmap.io instructions

## 1. setup colab VM

### a. notebook swift ssh
* open notebook swift ssh on colab, 
* make sure runtime type is gpu
* run notebook up to ssh tunnel cell

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

### b. ssh into colabp
```
ssh colabp
```

### c. (optional) run tmux in iterm
```
tmux -CC
```

### d. run Lang2motion script
```
cd /content/language2motion.gt
swift run -c release Lang2motion
```

### e. (optional) set git user name & email
```
git config --global user.name "Wojtek Czarnowski"
git config --global user.email "wojtek.czarnowski@gmail.com"
```

### f. (optional) scp rsa key into colabp
```
scp ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local colabp:/root/.ssh
```

### g. (optional) load github credentials
```
chmod 600 ~/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
ssh-agent bash
ssh-add /root/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
```

## 3. vscode integration

### 0. install extension
- Remote-SSH

### a. Remote-SSH: Connect to Host... ```colabp```

### b. install extensions (in container)
- CodeLLDB
- Maintained Swift Development Environment

### c. File/Open workspace...
```
/content/language2motion.gt/l2m.code-workspace
```

## 4. (optional) tensorbord

### a. start tensorboard
```
cd /content/language2motion.gt/data
tensorboard --logdir tboard
```
