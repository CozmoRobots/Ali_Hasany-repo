This file provides the necessary instructions to make it possible for Cozmo to recognize all the playing cards
including the Joker.

STEP 1:
Do a pip install {package name} incase you do not have all the necessary python packages
mentioned in Requirements.txt

1) move to your desired directory on your local machine and clone this repository
  >>git clone https://github.com/CozmoRobots/Ali_Hasany-repo.git
2) switch to the master branch
  >>git checkout master

STEP 2: Training the Machine Learning Model

I used the computer at my lab to train this ML model 
without the aid of any GPUs as the computer at our lab is a very powerful one
but runs a Linux operating system, particularly Ubuntu. Therefore I would recommend to
do the training on Ubuntu or some Linux variant but should work without a hitch on any
Windows machine too given you have installed everything as mentioned in Requiremets.txt.


1) move to the parent directory
  >>cd MLproject
2) rename the directory 'card_dataset' to 'train' for Ubuntu users
  >>mv card_dataset train
For Windows users just rename it conventionally or if you're using Powershell type the following
  >>Rename-Item -Path "card_dataset" -NewName "train"
3) start training
  >>python train.py
If your code throws GPU errors do the following
  >>CUDA_VISIBLE_DEVICES= python train.py
For Windows users/ Powershell:
  >>$env:CUDA_VISIBLE_DEVICES= ""; python train.py
4) After training is finished you will see the file "best_mode_4.h5" in the parent directory

STEP 3: Playing Blackjack with Cozmo
1) move the file 'best_model_4.h5' to the 'cozmo_BJ_sim' directory. Stay in the parent directory when issuing the following command!
  >>cp best_model_4.h5 ./cozmo_BJ_sim/



