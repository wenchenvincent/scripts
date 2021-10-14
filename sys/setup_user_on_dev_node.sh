## This script is to create a user in a dev node and setup the necessary groups and settings

# Create user and home directory
sudo useradd -m  wenchen
# Some machines do not have the group render
sudo usermod -aG sudo,video,docker wenchen || sudo usermod -aG sudo,video,render,docker wenchen
sudo passwd wenchen
sudo usermod --shell /bin/bash wenchen 

# Switch to user wenchen
su wenchen
cd /home/wenchen
mkdir dockerx
chmod a+rwx dockerx
# Copy over custom settings. More to be added
scp -P 30026 wenchen@10.216.64.100:~/.bash_aliases .
