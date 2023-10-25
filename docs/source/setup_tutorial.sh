#!/bin/sh
# this records setup for tutorial on nectar instances
#
# 1. Install all necessary system packages
# 2. Install all custom software into /usr/local (as ec2-user), if necessary
# 3. Perform rough tests as ec2-user
# 4. Create a new user (test-user)
# 5. Install all local user packages as (test-user)
# 6. Move all data into home of test-user
# 7. Make all changes to test-user’s bash profile/bashrc
# 8. As ec2-user: tar -cvzf home.tar.gz /home/test-user/ 
# 9. As root: cd /etc/skel && tar -xvzf /home/ec2-user/home.tar.gz 
# 10. Take an image (1st cut)
#
# — Now you can test the machine
# Testing: 
# 1. Log in as test-user
# 2. Run the workshop material, note down all changes to software packages that need to be made to fix it. 
#
# - Create a new machine from 1st cut
# 1. Log in as ec2-user, make the necessary changes, delete /etc/skel/* and use step 8/9 to save home to skel
#
# — Take an image (2nd cut) 
# Now you can test and iterate if necessary. 
#

## install mamba

#  wget may not exist
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

#  accept all defaults
bash Miniforge3-$(uname)-$(uname -m).sh

#  reload shell
. ~/.bashrc

#  install the package into a conda environment
mamba create -n gnlp -y -c conda-forge -c tyronechen python==3.9 genomenlp && mamba activate gnlp

# test key components (should print to stdout)
tokenise_bio -h
create_dataset_bio -h
sweep -h
train -h
cross_validate -h
interpret -h
fit_powerlaw -h

# setup dir
mkdir data src results

# download test data
cd data
curl -L -O "https://raw.githubusercontent.com/khanhlee/bert-promoter/main/data/non_promoter.fasta"
curl -L -O "https://raw.githubusercontent.com/khanhlee/bert-promoter/main/data/promoter.fasta"

# create test user for rollout (you need root access)
sudo useradd test-user
sudo -u test-user bash

# run through the pipeline following the online tutorial
