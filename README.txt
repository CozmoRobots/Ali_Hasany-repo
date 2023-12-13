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

STEP 3: Copying trained model to relevant directory
1) move the file 'best_model_4.h5' to the 'cozmo_BJ_sim' directory. Stay in the parent directory when issuing the following command!
  >>cp best_model_4.h5 ./cozmo_BJ_sim/
For Windows users just copy and paste the file or if using Powershell issue the following command:
  >>Copy-Item -Path "best_model_4.h5" -Destination ".\cozmo_BJ_sim\"
2) Now move to the 'cozmo_BJ_sim' directory
  >>cd cozmo_BJ_sim
3) Open the file 'cozmo_as_player.py'in any text editor of your choice
if you have VS code issue the following command:
  >>code -n cozmo_as_player.py
in line 37 change the following line:
model = load_model('best_model_2.h5')
to 
model = load_model('best_model_4.h5')

If you do not make the changes in the file above as mentioned it will simply use the model that I trained earlier

STEP 4: Playing Blackjack with Cozmo!
1) run the file 'cozmo_as_player.py'
>>python cozmo_as_player.py



