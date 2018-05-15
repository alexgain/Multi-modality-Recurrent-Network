# Multi-modality-Recurrent-Network
Novel arbitrarily recurrent structure that learns optimal pathways/adjacency matrix through a clever use of backpropagation.
Project for Machine Learning: Deep Learning course at JHU CS Department.

The GAN dateset images generated can be found at https://drive.google.com/file/d/1HLGoAPBRgXZ0z9TzKmmMglmqjvi1bS9u/view?usp=sharing. GAN dataset images in .npy format can be found at https://drive.google.com/file/d/1pXXLASH0dTjHSTuyG6wNAxqWQ7zG48vg/view.  Reference code for generating the images can be found here: https://github.com/robbiebarrat/art-DCGAN

Latest update:
* Trained on multiple datasets
* Trained via EA
* Trained via Synthetic images from GAN (Update: better results with architecture from main_gan.py)
* Adjacency matrix plots
* main_cifar is for cifar10 dataset, which a used slightly different loss and setup
* main multi (current) is for all other datasets (mnist, fashion-mnist, GAN dataset) for the main model
* main_fashion EA.py is for Evolutionary Algorithm training of the adjacency for fashion-mnist dataset.
* main_gan.py has a slightly specialized architecture for the GAN dataset that achieves better results than before.
