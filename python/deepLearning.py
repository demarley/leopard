"""
Created:        16 August 2018
Last Updated:   16 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Base class for performing deep learning 

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

from deepLearningPlotter import DeepLearningPlotter

import uproot
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as tf
from torch.autograd import Variable

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_curve, auc


# fix random seed for reproducibility
seed = 2018
np.random.seed(seed)


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
    


class DeepLearning(object):
    """Deep Learning base class"""
    def __init__(self):
        self.date = datetime.date.today().strftime('%d%b%Y')

        ## Handling NN objects and data -- set in the class
        self.df  = None          # dataframe containing physics information
        self.fpr = None          # ROC curve: false positive rate
        self.tpr = None          # ROC curve: true positive rate
        self.model = None      # Keras model
        self.accuracy  = {'mean':0,'std':0}   # k-fold accuracies
        self.histories = []           # model history (for ecah k-fold)
        self.train_data = {}          # set later
        self.test_data  = {}          # set later
        self.train_predictions = []   # set later
        self.test_predictions  = []   # set later

        ## Config options
        self.treename   = 'features'    # Name of TTree to access in ROOT file (via uproot)
        self.useLWTNN   = True          # export (& load model from) files for LWTNN
        self.dnn_name   = "dnn"         # name to access in lwtnn ('variables.json')
        self.hep_data   = ""            # Name for loading features (physics data) -- assumes all data in one file
        self.model_name = ""            # Name for saving/loading model
        self.output_dir = 'data/dnn/'   # directory for storing NN data
        self.dnn_method = None          # DNN method applied: classification/regression: ['binary','multi','regression']
        self.runDiagnostics = True      # Make plots pre/post training
        self.verbose_level  = 'INFO'
        self.verbose = False

        ## PyTorch objects
        self.loss_fn   = None  # pytorch loss function
        self.torch_opt = None  # pytorch optimizer

        ## NN architecture (from config)
        self.loss    = 'binary_crossentropy' # preferred for binary classification
        self.init    = 'normal'
        self.nNodes  = []
        self.dropout = None
        self.metrics = ['accuracy']
        self.features   = []
        self.epochs     = 1        
        self.optimizer  = 'adam'
        self.input_dim  = 1                  # len(self.features)
        self.output_dim = 1                  # number of output dimensions (# of categories/# of predictions for regression)
        self.batch_size = 32
        self.activations   = ['elu']
        self.kfold_splits  = 2
        self.nHiddenLayers = 1
        self.learning_rate = 1e-4
        self.earlystopping = {}              # {'monitor':'loss','min_delta':0.0001,'patience':5,'mode':'auto'}


    def initialize(self):   #,config):
        """Initialize a few parameters after they've been set by user"""
        self.msg_svc       = util.VERBOSE()
        self.msg_svc.level = self.verbose_level
        self.msg_svc.initialize()
        self.verbose = not self.msg_svc.compare(self.verbose_level,"WARNING") # verbose if level is <"WARNING"

        # Set name for the model, if needed
        if not self.model_name:
            self.model_name = self.hep_data.split('/')[-1].split('.')[0]+'_'+self.date

        # initialize empty dictionaries, lists
        self.test_data  = {'X':[],'Y':[]}
        self.train_data = {'X':[],'Y':[]}
        self.test_predictions  = []
        self.train_predictions = []

        self.fpr = []  # false positive rate
        self.tpr = []  # true positive rate
        self.histories  = []


        ## -- Plotting framework
        print " >> Store output in ",self.output_dir
        self.plotter = DeepLearningPlotter()  # class for plotting relevant NN information
        self.plotter.output_dir   = self.output_dir
        self.plotter.image_format = 'pdf'

        ## -- Adjust model architecture parameters (flexibilty in config file)
        if len(self.nNodes)==1 and self.nHiddenLayers>0:
            # All layers (initial & hidden) have the same number of nodes
            self.msg_svc.DEBUG("DL : Setting all layers ({0}) to have the same number of nodes ({1})".format(self.nHiddenLayers+1,self.nNodes))
            nodes_per_layer = self.nNodes[0]
            self.nNodes = [nodes_per_layer for _ in range(self.nHiddenLayers+1)] # 1st layer + nHiddenLayers

        ## -- Adjust activation function parameter (flexibilty in config file)
        if len(self.activations)==1:
            # Assume the same activation function for all layers (input,hidden,output)
            self.msg_svc.DEBUG("DL : Setting input, hidden, and output layers ({0}) \n".format(self.nHiddenLayers+2)+\
                               "     to have the same activation function {0}".format(self.activations[0]) )
            activation = self.activations[0]
            self.activations = [activation for _ in range(self.nHiddenLayers+2)] # 1st layer + nHiddenLayers + output
        elif len(self.activations)==2 and self.nHiddenLayers>0:
            # Assume the last activation is for the output and the first+hidden layers have the first activation
            self.msg_svc.DEBUG("DL : Setting input and hidden layers ({0}) to the same activation function, {1},\n".format(self.nHiddenLayers+1,self.activations[0])+\
                               "     and the output activation to {0}".format(self.activations[1]) )
            first_hidden_act = self.activations[0]
            output_act       = self.activations[1]
            self.activations = [first_hidden_act for _ in range(self.nHiddenLayers+1)]+[output_act]

        return


    ## Single functions to run all of the necessary pieces
    def training(self):
        """Train NN model"""
        self.load_hep_data()
        self.build_model()

        # hard-coded :/
        target_names  = ["bckg","signal"]
        target_values = [0,1]
        self.plotter.initialize(self.df,target_names,target_values)

        if self.runDiagnostics:
            self.diagnostics(preTraining=True)     # save plots of the features and model architecture

        self.msg_svc.INFO("DL : Train the model")
        self.train_model()

        self.msg_svc.INFO("DL : SAVE MODEL")
        self.save_model()

        if self.runDiagnostics:
            self.diagnostics(postTraining=True)    # save plots of the performance in training/testing

        return


    def inference(self,data=None):
        """
        Run inference of the NN model
        User responsible for diagnostics if not doing training: 
        -> save all predictions (& labels) using 'self.test_predictions'
           then call individual functions:
              plot_features()   -> compare features of the inputs
              plot_prediction() -> compare output prediction (works for classification)
              plot_ROC()        -> signal vs background efficiency (need self.fpr, self.tpr filled)
        """
        self.msg_svc.INFO("DL : Load model for inference")
        self.load_model()

        self.msg_svc.INFO("DL : Load data")
        if data is None:
            try:
                self.load_hep_data()
                data = self.df[self.features]
            except:
                self.msg_svc.ERROR("DL : inference() cannot proceed because 'data' is None and cannot load HEP data")
                self.msg_svc.ERROR("DL : Please check your implementation.")
                return -999

        self.msg_svc.INFO("DL : Make inference")
        prediction = self.predict(data)

        return prediction


    ## Specific functions to perform training/inference tasks
    def build_model(self):
        """Construct the NN model -- only Keras support for now"""
        self.msg_svc.INFO("DL : Build the neural network model")

        ## Declare the model
        layers = []
        layers.append( {'in':int(self.input_dim),'out':int(self.nNodes[0])} )
        for i,n in enumerate(self.nNodes):
            if i==len(self.nNodes)-1: continue
            layers.append( {'in':int(n),'out':int(self.nNodes[i+1])} )
        layers.append( {'in':int(self.nNodes[-1]),'out':self.output_dim} )

        self.model = LeopardNet(layers)

        self.loss_fn   = torch.nn.BCELoss()
        self.torch_opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) #1e-4)

        return


    def train_epoch(self,X,Y):
        """"""
        losses = []
        for beg_i in range(0, len(X), self.batch_size):
            x_batch = torch.from_numpy(X[beg_i:beg_i+self.batch_size,:])
            y_batch = torch.from_numpy(Y[beg_i:beg_i+self.batch_size])
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch).float().unsqueeze_(-1)  # modify dimensions (X,) -> (X,1)

            self.torch_opt.zero_grad()

            y_hat = self.model(x_batch)         # forward
            loss = self.loss_fn(y_hat, y_batch) # compute loss
            loss.backward()                     # compute gradients
            self.torch_opt.step()               # update weights

            losses.append(loss.data.numpy())

        return losses



    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        self.msg_svc.INFO("DL : Train the model!")

        X = self.df[self.features].values
        Y = self.df['target'].values

        kfold = StratifiedKFold(n_splits=self.kfold_splits, shuffle=True, random_state=seed)
        nsplits = kfold.get_n_splits(X,Y)
        cvpredictions = []                 # compare outputs from each cross-validation

        self.msg_svc.INFO("DL :   Fitting K-Fold cross validation".format(self.kfold_splits))
        for ind,(train,test) in enumerate(kfold.split(X,Y)):
            self.msg_svc.DEBUG("DL :   - Fitting K-Fold {0}".format(ind))

            Y_train = Y[train]
            Y_test  = Y[test]

            # -- store test/train data from each k-fold to compare later
            self.test_data['X'].append(X[test])
            self.test_data['Y'].append(Y_test)
            self.train_data['X'].append(X[train])
            self.train_data['Y'].append(Y_train)

            ## Fit the model to training data & save the history
            self.model.train()
            e_losses = []
            for t in range(self.epochs):
                e_losses += self.train_epoch(X[train],Y_train)
                self.msg_svc.INFO("DL :    Epoch {0} -- Loss {1}".format(t,e_losses[-1]))
            self.histories.append(e_losses)

            # evaluate the model
            self.msg_svc.DEBUG("DL :     + Evaluate the model: ")
            self.model.eval()

            # Evaluate training sample
            self.msg_svc.INFO("DL : Predictions from training sample")
            train_predictions = self.predict(X[train])
            self.train_predictions.append( train_predictions )

            # Evaluate test sample
            self.msg_svc.INFO("DL : Predictions from testing sample")
            test_predictions  = self.predict(X[test])
            self.test_predictions.append( test_predictions )

            # Make ROC curve from test sample
            self.msg_svc.INFO("DL : Make ROC curves")
            fpr,tpr,_ = roc_curve( Y[test], test_predictions )
            self.fpr.append(fpr)
            self.tpr.append(tpr)

        self.msg_svc.INFO("DL :   Finished K-Fold cross-validation: ")
        self.accuracy = {'mean':np.mean(cvpredictions),'std':np.std(cvpredictions)}
        self.msg_svc.INFO("DL :   - Accuracy: {0:.2f}% (+/- {1:.2f}%)".format(np.mean(cvpredictions), np.std(cvpredictions)))

        return


    def predict(self,data=None):
        """Return the prediction from a test sample"""
        self.msg_svc.DEBUG("DL : Get the DNN prediction")
        if data is None:
            self.msg_svc.ERROR("DL : predict() given NoneType data. Returning -999.")
            return -999.
        data = torch.from_numpy(data)
        return self.model( Variable(data) )


    def load_hep_data(self,variables2plot=[]):
        """
        Load the physics data (flat ntuple) for NN using uproot
        Convert to DataFrame for easier slicing 

        @param variables2plot    If there are extra variables to plot, 
                                 that aren't features of the NN, include them here
        """
        file    = uproot.open(self.hep_data)
        data    = file[self.treename]
        self.df = data.pandas.df( self.features+['target']+variables2plot )

        self.metadata = file['metadata']   # names of samples, target values, etc.

        return


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


    def diagnostics(self,preTraining=False,postTraining=False):
        """Diagnostic tests of the NN"""

        self.msg_svc.INFO("DL : Diagnostics")

        # Diagnostics before the training
        if preTraining:
            self.msg_svc.INFO("DL : -- pre-training")
            self.plotter.features()                        # compare features for different targets
            self.plotter.correlation()                     # correlations of features

        # post training/testing
        if postTraining:
            self.msg_svc.INFO("DL : -- post-training")

            self.msg_svc.INFO("DL : -- post-training :: PREDICTIONS ")
            train = {'X':self.train_predictions,'Y':self.train_data['Y']}
            test  = {'X':self.test_predictions,'Y':self.test_data['Y']}
            self.plotter.prediction(train,test)   # compare DNN prediction for different targets

            self.msg_svc.INFO("DL : -- post-training :: ROC")
            self.plotter.ROC(self.fpr,self.tpr,self.accuracy)  # ROC curve for signal vs background
            self.msg_svc.INFO("DL : -- post-training :: History")
            self.plotter.loss_history(self.histories) # loss as a function of epoch

        return


## THE END ##
