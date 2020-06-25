# 1. start instance
```
gcloud compute instances start fastai-1
```
# 2. put ip (34.90.209.166) in ~/.ssh/config twice
# 3. ssh and set up tunnels
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
# 6. start tensorboard
```
conda activate py38
cd language2motion.gt/data/
tensorboard --port 8082 --logdir tboard/Motion2label/
```
# 7. on second tmux tab: update language2motion sources
# a. bash inside docker container
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
# 8. vscode
## a. set docker.host to 
```
ssh://fastai-1-vscode
```
## b. attach to running container...
## c. open workspace

# ports
* 8080 - jupyter lab 1
* 8081 - jupyter lab 2, in swift-jupyter docker container
* 8082 - tensorboard (in swift-jupyter docker container?)
```
gcloud compute ssh jupyter@fastai-1 -- -L 8080:localhost:8080 -L 8081:localhost:8081 -L 8082:localhost:8082
```

```
gcloud compute instances stop fastai-1
```
