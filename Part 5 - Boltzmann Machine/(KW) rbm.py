# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:49:55 2019

@author: kuanw
"""

# Boltzmann Machines
# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') # 'delminiter' same as 'sep' above, '\t' means seperated by tab
training_set = np.array(training_set, dtype =  'int64')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int64')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = [] # list of lists used here (instead of np arrays) as we are converting them into torch tensors later
    for id_users in range(1, nb_users+1): # loop through all users
        id_movies = data[:, 1][data[:, 0] == id_users] # to extract id_movies based on id_users
        id_ratings = data[:, 2][data[:, 0] == id_users] # to extract id_ratings based on id_users
        ratings = np.zeros(nb_movies) # create numpy array of zeros (of ratings) corresponding to nb_movies
        ratings[id_movies-1] = id_ratings # replace list of zeros with id_ratings, if the users gave that id_movie a rating
        new_data.append(list(ratings)) #convert the ratings to list, and append it to list
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
        
# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into  binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # weight - randomly initialise tensor of size (nh, nv) according to a normal distribution (mean zero, variance one)
        self.a = torch.randn(1, nh) # bias for probability of hidden node, given visible node (ph given v) - 2D tensor with 1st Dimension as Batch, 2nd Dimention as the Bias
        self.b = torch.randn(1, nv) # bias for probability of visible node, given hidden node (pv given h)
    # Gibs sampling to approximate log likelihood gradient
    # 1) compute probability of hidden nodes, given the visible nodes (ph given v is the sigmoids activation function)
    # 2) once we have 1) we can sample the activations of the hidden nodes
    def sample_h(self, x): # x will correspond to the visible neurons, v (in the probabilities, p h given v)
        wx = torch.mm(x, self.W.t()) # assign weights to the respective visible neurons
        activation = wx + self.a.expand_as(wx) # a.expand_as(wx) to expand the dimension of bias to apply it to each line of the mini batch
        p_h_given_v = torch.sigmoid(activation) # p_h_given_v is vector containing the probabilities the hidden nodes are activated, given the values of the visible note (i.e. ratings of the users)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # torch.bernoulli(p_h_given_v) activate the hidden note based on p_h_given_v value. We take a random number between 0 and 1, if random number is below p_h_given_v, we will activate the neuron and vice versa
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # do not need to take the transpose
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h) # torch.bernoulli(p_v_given_h) activate the hidden note based on p_v_given_h value. We take a random number between 0 and 1, if random number is below p_h_given_v, we will assign the visible note a value of 1 (i.e. predict the user Like the movie) and vice versa
    # k-step contrastive divergence (to approximate RBM log-likelihood gradient)
    # 1) RBM can be seen as a probalilistic graphical model
    # 2) The goal would be to maximise the log-likelihood of the training set
    # 3) We will need to compute the gradient to maximise the log likelihood
    # 4) Since computation of gradient is too heavy, we will approximate using k-step contrastive divergence
        # - Gibbs Chain created by sampling K times the hidden nodes and visible nodes
            # a) start with input vector v0, then based on the probabilities ph given v0,
            # b) we sample the first hidden nodes, then we take these sample hidden nodes h1 as inputs,
            # c) we sample the visible nodes with the probabilities pv given h1, then we use these sample visible nodes
            # d) to sample again the hidden nodes with the probabilities ph given v1 
            # e) we repeat this process k times
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0) # ", 0" to keep as 2 dimensions
        self.a += torch.sum((ph0 - phk), 0)
    def predict(self, x): # x: visible nodes
        _, h = self.sample_h(x)
        _, v = self.sample_v(h)
        return v
      
nv = len(training_set[0]) # number of movies
nh = 100 # 100 features to detect, tune to improve the model
batch_size = 100 # batch size of 100, tune to improve the model
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # counter, increment after each epoch, for nomalizing the train loss (train loss/s)
    for id_user in range(0, nb_users - batch_size, 100):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0) # returns ph0 given v0 which is the original ratings of the movies for all the users of our batch
        for k in range(10): # 10 steps for k-step for contrastive convergence
            # continue for 10th round, of sampling hidden then visible then hidden nodes...
            _,hk = rbm.sample_h(vk) # use vk instead of v0, we do not change v0 as we need it to compute our loss
            _,vk = rbm.sample_v(hk) 
            vk[v0<0] = v0[v0<0] # ensure training does not affect ratings that have no values (i.e. freezing ratings with -1)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>0]-vk[v0>0])) # updating the train loss
        s += 1
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s = 0. # counter, increment after each epoch
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] # training set is the input that will be used to activate the hidden neurons to get the output
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0: # if there are at least some ratings that are existent, we can make some predictions
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>0]-v[vt>0]))
        s += 1
print('test loss: '+str(test_loss/s))

# Making a prediction
predictions = rbm.predict(training_set)
predictions = torch.Tensor.numpy(predictions)
print(pd.DataFrame(predictions))