# 1. start instance
```
gcloud compute instances start fastai-1
```

# 2. set up ssh tunnels, put ip (34.90.135.237) in ~/.ssh/config twice

ports:
* 8080 - jupyter lab 1
* 8081 - jupyter lab 2, in swift-jupyter docker container
* 6006 - tensorboard, in swift-jupyter docker container

```
Host fastai-1
HostName 34.90.135.237
StrictHostKeyChecking no
User jupyter
Port 22
LocalForward 8080 localhost:8080
LocalForward 8081 localhost:8081
LocalForward 6006 localhost:6006
```

```
Host fastai-1-vscode
HostName 34.90.135.237
StrictHostKeyChecking no
User jupyter
Port 22
```

# 3. ssh and activate tunnels
```
ssh fastai-1
```

# 4. start tmux
```
tmux -CC
```

# 5. start swift-jupyter container
```
docker start swift-jupyter
```

# 6. start tensorboard (inside docker container)
```
docker exec -it swift-jupyter bash
cd language2motion.gt/data/
tensorboard --bind_all --logdir runs/Lang2motion/run_4/
```
# 7. on second tmux tab: update language2motion sources
# a. exec into docker container
```
docker exec -it swift-jupyter bash
```
# b. git pull
```
ssh-agent bash
ssh-add /notebooks/.ssh/id_rsa_wcz\@MacBook-Wojtka.local
cd language2motion.gt/
git pull
```
# 8. start vscode
## a. set docker.host to 
```
ssh://fastai-1-vscode
```
## b. Remote-Containers: Attach to Running Container...
```
/swift-jupyter
```
## c. File/Open Workspace...
```
/notebooks/language2motion.gt/l2m.code-workspace
```
# stop instance
```
gcloud compute instances stop fastai-1
```
