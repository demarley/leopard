"""
Created:        16 August 2018
Last Updated:   16 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Class for performing deep learning in pytorch

Designed for running on desktop at TAMU
with specific set of software installed
--> not guaranteed to work in CMSSW environment!

Does not use ROOT directly.
Instead, this is setup to use flat ntuples
that are accessed via uproot.

> UPROOT:     https://github.com/scikit-hep/uproot
> KERAS:      https://keras.io/
> TENSORFLOW: https://www.tensorflow.org/
> PYTORCH:    http://pytorch.org/
> LWTNN:      https://github.com/lwtnn/lwtnn
"""
import json
import util
import datetime
import collections

from deepLearning import DeepLearning

import uproot
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as tf
from torch.autograd import Variable

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve


class LeopardNet(nn.Module):
    """Neural Network for Leopard in PyTorch
       Adapted from (16 August 2018)
         https://github.com/thongonary/surf18-tutorial/blob/master/tuto-8-torch.ipynb
    """
    def __init__(self,layers):
        super(LeopardNet,self).__init__()
        self.dense = nn.ModuleList()
        for l,layer in enumerate(layers):
            self.dense.append( nn.Linear(layer['in'],layer['out']) )
    
    def forward(self, x): 
        """All the computation steps of the input are defined in this function"""
        nlayers = len(self.dense)
        for i,d in enumerate(self.dense):
            x = d(x)
            x = tf.relu(x) if i!=nlayers-1 else tf.sigmoid(x)
        return x



class DeepLearningTorch(DeepLearning):
    """Deep Learning pytorch class"""
    def __init__(self):
        DeepLearning.__init__(self)

        ## PyTorch objects
        self.loss_fn   = None  # pytorch loss function
        self.torch_opt = None  # pytorch optimizer

    def initialize(self):   #,config):
        """Initialize a few parameters after they've been set by user"""
        DeepLearning.initialize(self)
        return


    ## Specific functions to perform training/inference tasks
    def build_model(self):
        """Construct the NN model -- only Keras support for now"""
        self.msg_svc.INFO("DLPYTORCH : Build the neural network model")

        ## Declare the model
        layers = []
        layers.append( {'in':int(self.input_dim),'out':int(self.nNodes[0])} )
        for i,n in enumerate(self.nNodes):
            if i==len(self.nNodes)-1: continue
            layers.append( {'in':int(n),'out':int(self.nNodes[i+1])} )
        layers.append( {'in':int(self.nNodes[-1]),'out':self.output_dim} )

        self.model = LeopardNet(layers)
        self.model.cuda()

        self.loss_fn   = torch.nn.BCELoss()
        self.torch_opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) #1e-4)

        return


    def train_epoch(self,X,Y):
        """"""
        losses = []
        for beg_i in range(0, len(X), self.batch_size):
            x_batch = torch.from_numpy(X[beg_i:beg_i+self.batch_size,:])
            y_batch = torch.from_numpy(Y[beg_i:beg_i+self.batch_size])
            x_batch = Variable(x_batch).cuda()
            y_batch = Variable(y_batch).float().unsqueeze_(-1).cuda()  # modify dimensions (X,) -> (X,1)

            self.torch_opt.zero_grad()

            y_hat = self.model(x_batch)         # forward
            loss = self.loss_fn(y_hat, y_batch) # compute loss
            loss.backward()                     # compute gradients
            self.torch_opt.step()               # update weights

            losses.append(loss.data.cpu().numpy())

        return losses



    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        X = self.df[self.features].values
        Y = self.df['target'].values

        kfold = StratifiedKFold(n_splits=self.kfold_splits, shuffle=True, random_state=seed)
        nsplits = kfold.get_n_splits(X,Y)
        cvpredictions = []                 # compare outputs from each cross-validation

        self.msg_svc.INFO("DLPYTORCH :   Fitting K-Fold cross validations")
        for ind,(train,test) in enumerate(kfold.split(X,Y)):
            self.msg_svc.INFO("DLPYTORCH :   - Fitting K-Fold {0}".format(ind))

            Y_train = Y[train]
            Y_test  = Y[test]

            # -- store test/train data from each k-fold as histograms (to compare later)
            h_tests  = {}
            h_trains = {}
            for n,v in self.targets.iteritems():
                h_tests[n]  = ROOT.TH1D("test_"+n,"test_"+n,10,0,10)
                h_trains[n] = ROOT.TH1D("train_"+n,"train_"+n,10,0,10)

            # fill histogram for each target
            for (n,v) in enumerate(self.targets.iteritems()):
                [h_tests[n].Fill(i)  for i in X[test][np.where(Y_test==v)]]
                [h_trains[n].Fill(i) for i in X[train][np.where(Y_train==v)]]


            ## Fit the model to training data & save the history
            self.model.train()
            e_losses = []
            for t in range(self.epochs):
                e_losses += self.train_epoch(X[train],Y_train)
                self.msg_svc.INFO("DLPYTORCH :    Epoch {0} -- Loss {1}".format(t,e_losses[-1]))
            self.histories.append(e_losses)

            # evaluate the model
            self.msg_svc.DEBUG("DLPYTORCH : Evaluate the model: ")
            self.model.eval()

            # Evaluate training sample
            self.msg_svc.INFO("DLPYTORCH : Predictions from training sample")
            train_predictions = self.predict(X[train])
            self.train_predictions.append(train_predictions)

            # Evaluate test sample
            self.msg_svc.INFO("DLPYTORCH : Predictions from testing sample")
            test_predictions  = self.predict(X[test])
            self.test_predictions.append(test_predictions)

            # Make ROC curve from test sample
            self.msg_svc.INFO("DLPYTORCH : Make ROC curves")
            fpr,tpr,_ = roc_curve(Y[test], test_predictions)
            self.fpr.append(fpr)
            self.tpr.append(tpr)

            # Plot the predictions to compare test/train
            self.msg_svc.INFO("DLPYTORCH : Plot the train/test predictions")
            self.plotter.prediction(h_trains,h_tests)   # compare DNN prediction for different targets

        self.msg_svc.INFO("DLPYTORCH :   Finished K-Fold cross-validation: ")
        self.accuracy = {'mean':np.mean(cvpredictions),'std':np.std(cvpredictions)}
        self.msg_svc.INFO("DLPYTORCH :   - Accuracy: {0:.2f}% (+/- {1:.2f}%)".format(np.mean(cvpredictions), np.std(cvpredictions)))

        return


    def predict(self,data=None):
        """Return the prediction from a test sample"""
        self.msg_svc.DEBUG("DLPYTORCH : Get the DNN prediction")
        if data is None:
            self.msg_svc.ERROR("DLPYTORCH : predict() given NoneType data. Returning -999.")
            return -999.
        data = torch.from_numpy(data)

        return self.model( Variable(data,volatile=True).cuda() )

    def load_model(self,from_lwtnn=False):
        """Load existing model to make plots or predictions"""
        output = self.output_dir+'/'+self.model_name
        self.model.load_state_dict(torch.load(output))
        self.model.eval()
        return

    def save_model(self,to_lwtnn=False):
        """Save the model for use later"""
        output = self.output_dir+'/'+self.model_name
        torch.save(self.model.state_dict(),output)
        return


## THE END ##
