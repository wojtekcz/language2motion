apt-get update
apt-get install -y openssh-server autossh tmux mc htop
pip2 install glances google-auth-oauthlib==0.4.1 grpcio==1.24.3

baseURL=https://raw.githubusercontent.com/wojtekcz/language2motion/master/notebooks/Colab/

# cp sshd_config /etc/ssh/sshd_config
wget -nv $baseURL/sshd_config -O /etc/ssh/sshd_config

/etc/init.d/ssh restart
/etc/init.d/ssh status

# cp bashrc /root/.bashrc
wget -nv $baseURL/bashrc -O /root/.bashrc

mkdir -p /root/.ssh
chmod 700 /root/.ssh

# cp authorized_keys /root/.ssh/authorized_keys
wget -nv $baseURL/authorized_keys -O /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

privateKeyPath=/root/.ssh/private_key.pem
# cp private_key.pem $privateKeyPath
wget -nv $baseURL/private_key.pem -O $privateKeyPath
chmod 600 $privateKeyPath

options="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null"
ssh $options -i $privateKeyPath -f -R 51552:localhost:22 wojtekcz.first@wojtekcz-51552.portmap.io -N -v &
