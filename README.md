# PyTorch UCL Team Classifier

## Introduction
CNN Classifier that I realized to exercise on Deep Learning, PyTorch and Transfer Learning.
This program recognizes pictures of players belonging to different UCL teams.

Disclaimer: this is mostly a didactic project I created in order to get familiar with the most popular deep learning tools and technologies, the results I've obtained are to be interpreted in the context of a didactic experiment. 

## Setup
The dataset was created by me manually by scraping google images and gathering pictures containing players from the teams taking part in the UCL, I collected a total of 100 images for 8 teams, for a total of 800 images. 


I decided to run my model on Google Colab in order to exploit the free GPU usage to train my model faster, consequently all the data is stored on my Google Drive account

## Quickstart

You can use conda to install all the required libraries

    conda env create -f environment.yml

Activate the environment:

    conda activate pytorch

to train the model:

    python train.py

to run a test of prediction on some test images:

    python predict.py


## Deep Learning Architecture

Since the amount of data at my disposal was relatively low, I decided to use the technique of Transfer Learning and to use the Resnet model as a fixed feature extractor for the images. 

This way I managed to get good results while at the same time reducing the training time to a minimun.  


## Results

With this model I was able to obtain an accuracy of 87% on my test set, which is a pretty good result considering the limited amount of images that the dataset were composed of. 
