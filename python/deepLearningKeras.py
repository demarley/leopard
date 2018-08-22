"""
Created:        16 August 2018
Last Updated:   16 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Class for performing deep learning in keras

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
import sys
import json
import util
import datetime
import collections

from deepLearning import DeepLearning

import ROOT
import uproot
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential,model_from_json,load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_curve




class DeepLearningKeras(DeepLearning):
    """Deep Learning keras class"""
    def __init__(self):
        DeepLearning.__init__(self)


    def initialize(self):
        """Initialize a few parameters after they've been set by user"""
        DeepLearning.initialize(self)
        return


    ## Specific functions to perform training/inference tasks
    def build_model(self):
        """Construct the NN model -- only Keras support for now"""
        self.msg_svc.INFO("DLKERAS : Build the neural network model")

        ## Declare the model
        self.model = Sequential() # The Keras Sequential model is a linear stack of layers.

        ## Add 1st layer
        self.model.add( Dense( int(self.nNodes[0]), input_dim=self.input_dim, kernel_initializer=self.init, activation=self.activations[0]) )

        ## Add hidden layer(s)
        for h in range(self.nHiddenLayers):
            self.model.add( Dense( int(self.nNodes[h+1]), kernel_initializer=self.init, activation=self.activations[h+1]) )

        ## Add the output layer
        self.model.add( Dense(self.output_dim,kernel_initializer=self.init, activation=self.activations[-1]) )

        ## Build the model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.model.summary()

        return


    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        callbacks_list = []
        if self.earlystopping:
            earlystop = EarlyStopping(**self.earlystopping)
            callbacks_list = [earlystop]

        print " SIZE OF DF = ",sys.getsizeof(self.df)

        self.msg_svc.INFO("DLKERAS :   Fitting model to data ")
        X = self.df[self.features].values
        Y = self.df['target'].values

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.3)           # split into train/test
        Xtrain, Xval, Ytrain, Yval   = train_test_split(Xtrain,Ytrain,test_size=0.3) # split train into train/validation

        ## Fit the model to training data & save the history
        self.model.summary()
        history = self.model.fit(Xtrain,Ytrain,epochs=self.epochs,\
                                 batch_size=self.batch_size,verbose=self.verbose,validation_data=(Xval,Yval))
        print " save history"
        self.histories.append(history)

        # -- store test/train data from each k-fold as histograms (to compare later)
        h_tests  = {}
        h_trains = {}
        for n,v in self.targets.iteritems():
            h_tests[n]  = ROOT.TH1D("test_"+n,"test_"+n,10,0,1)
            h_trains[n] = ROOT.TH1D("train_"+n,"train_"+n,10,0,1)

        # Evaluate training sample
        self.msg_svc.INFO("DLKERAS : Predictions from training sample")
        train_predictions = self.predict(Xtrain)

        # Evaluate test sample
        self.msg_svc.INFO("DLKERAS : Predictions from testing sample")
        test_predictions = self.predict(Xtest)
        predictions = self.model.evaluate(Xtest, Ytest,verbose=self.verbose,batch_size=self.batch_size)
        cvpredictions.append(predictions[1] * 100)

        # Make ROC curve from test sample
        self.msg_svc.INFO("DLKERAS : Make ROC curves")
        fpr,tpr,_ = roc_curve(Ytest, test_predictions)
        self.fpr.append(fpr)
        self.tpr.append(tpr)

        # fill histogram for each target
        for (n,v) in enumerate(self.targets.iteritems()):
            [h_tests[n].Fill(i)  for i in test_predictions[np.where(Y_test==v)]]
            [h_trains[n].Fill(i) for i in train_predictions[np.where(Y_train==v)]]

        # Plot the predictions to compare test/train
        self.msg_svc.INFO("DLKERAS : Plot the train/test predictions")
        self.plotter.prediction(h_trains,h_tests)   # compare DNN prediction for different targets

        self.msg_svc.INFO("DLKERAS :   Finished fitting model ")
        self.msg_svc.INFO("DLKERAS :   - Accuracy: {0:.2f}% (+/- {1:.2f}%)".format(np.mean(cvpredictions), np.std(cvpredictions)))

        return


    def load_model(self,from_lwtnn=False):
        """Load existing model to make plots or predictions"""
        self.model = None
        if from_lwtnn:
            model_json = open(self.model_name+"_model.json",'r').read()
            self.model = model_from_json(model_json)
            self.model.load_weights(self.model_name+"_weights.h5")
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        else:
            self.model = load_model('{0}.h5'.format(self.model_name))

        return


    def save_model(self,to_lwtnn=False):
        """Save the model for use later"""
        output = self.output_dir+'/'+self.model_name
        if to_lwtnn:
            ## Save to format for LWTNN
            self.save_features()            ## Save variables to JSON file

            ## model architecture
            model_json = self.model.to_json()
            with open(output+'_model.json', 'w') as outfile:
                outfile.write(model_json)

            ## save the model weights
            self.model.save_weights(output+'_weights.h5')
        else:
            self.model.save('{0}.h5'.format(output))     # creates a HDF5 file of model

        return


    def predict(self,data=None):
        """Return the prediction from a test sample"""
        self.msg_svc.DEBUG("DLKERAS : Get the DNN prediction")
        if data is None:
            self.msg_svc.ERROR("DLKERAS : predict() given NoneType data. Returning -999.")
            return -999.

        return self.model.predict( data )


    def save_features(self):
        """
        Save the features to a json file to load via lwtnn later
        Hard-coded scale & offset; must change later if necessary
        """
        text = """  {
    "inputs": ["""

        for fe,feature in enumerate(self.features):
            comma = "," if fe!=len(self.features) else ""
            tmp = """
      {"name": "%(feature)s",
       "scale":  1,
       "offset": 0}%(comma)s""" % {'feature':feature,'comma':comma}
            text += tmp
        text += "],"
        text += """
    "class_labels": ["%(name)s"],
    "keras_version": "%(version)s",
    "miscellaneous": {}
  }
""" % {'version':keras.__version__,'name':self.dnn_name}

        varsFileName = self.output_dir+'/variables.json'
        varsFile     = open(varsFileName,'w')
        varsFile.write(text)

        return

## THE END ##
