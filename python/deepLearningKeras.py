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

from deepLearningPlotter import DeepLearningPlotter
from guppy import hpy

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
from sklearn.preprocessing import StandardScaler



class DeepLearningKeras(object):
    """Deep Learning keras class"""
    def __init__(self):
        self.date = datetime.date.today().strftime('%d%b%Y')

        # fix random state for reproducibility
        self.seed = 2018
        np.random.seed(self.seed)

        ## Handling NN objects and data -- set in the class
        self.df  = None          # dataframe containing physics information
        self.fpr = None          # ROC curve: false positive rate
        self.tpr = None          # ROC curve: true positive rate
        self.model = None      # Keras model
        self.accuracy  = {'mean':0,'std':0}   # k-fold accuracies
        self.histories = []           # model history (for ecah k-fold)
        self.train_data = {}          # set later
        self.test_data  = {}          # set later
        self.targets = collections.OrderedDict()

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


    def initialize(self):
        """Initialize a few parameters after they've been set by user"""
        self.msg_svc       = util.VERBOSE()
        self.msg_svc.level = self.verbose_level
        self.msg_svc.initialize()
        self.verbose = self.verbose_level=="DEBUG"

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
        self.msg_svc.INFO("DL :  >> Store output in {0}".format(self.output_dir))
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



    ## Specific functions to perform training/inference tasks
    def build_model(self):
        """Construct the NN model -- only Keras support for now"""
        self.msg_svc.INFO("DLKERAS : Build the neural network model")

       	print self.activations
       	print self.nNodes
       	print self.nHiddenLayers

        ## Declare the model
        self.model = Sequential() # The Keras Sequential model is a linear stack of layers.
        ## Add 1st layer
        self.model.add( Dense( int(self.nNodes[0]), input_dim=self.input_dim, kernel_initializer=self.init, activation=self.activations[0]) )
#        ## Add hidden layer(s)
#        for h in range(self.nHiddenLayers):
#            self.model.add( Dense( int(self.nNodes[h+1]), kernel_initializer=self.init, activation=self.activations[h+1]) )

        ## Add the output layer
        self.model.add( Dense(self.output_dim,kernel_initializer=self.init, activation=self.activations[-1]) )

        ## Build the model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.model.summary()

        return

    def training(self):
        """Train NN model"""
        self.load_hep_data()
        self.build_model()

        self.plotter.initialize(self.df,self.targets)

        if self.runDiagnostics:
            self.diagnostics(pre=True)     # save plots of the features and model architecture

        self.msg_svc.INFO("DL : Train the model")
        self.train_model()

        self.msg_svc.INFO("DL : SAVE MODEL")
        self.save_model()

        if self.runDiagnostics:
            self.diagnostics(post=True)    # save plots of the performance in training/testing

        return


    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        callbacks_list = []
        if self.earlystopping:
            earlystop = EarlyStopping(**self.earlystopping)
            callbacks_list = [earlystop]
        h = hpy() 
        print h.heap()
        print " SIZE OF DF = ",sys.getsizeof(self.df)
        print " SIZE OF MODEL = ",sys.getsizeof(self.model)

        self.msg_svc.INFO("DLKERAS :   Fitting model to data ")
        Xtrain = self.df[self.features].values
        Ytrain = self.df['target'].values

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain,Ytrain,test_size=0.3) # split into train/test
        print ' train '
        print Xtrain.dtype
        print Ytrain.dtype
        print '\n test '
        print Xtest.dtype
        print Ytest.dtype
        print
        print self.epochs
        print self.batch_size
        print self.verbose

        ## Fit the model to training data & save the history
        self.model.summary()
        print h.heap()
        print " ready to fit..."
        history = self.model.fit(Xtrain,Ytrain,epochs=5,batch_size=64,verbose=1,validation_split=0.25)
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



    def load_hep_data(self,variables2plot=[]):
        """
        Load the physics data (flat ntuple) for NN using uproot
        Convert to DataFrame for easier slicing 

        @param variables2plot    If there are extra variables to plot, 
                                 that aren't features of the NN, include them here
        """
        file = uproot.open(self.hep_data)
        data = file[self.treename]
        df   = data.pandas.df( self.features+['target']+variables2plot )

        self.msg_svc.DEBUG("DL : Scale the inputs")
        scaler = StandardScaler()
        df[self.features] = scaler.fit_transform(df[self.features])

        # Make the dataset sizes equal (trim away some background)
        fraction=0.001
        signal = df[ (df.target==1)&(df.AK4_CSVv2>=0) ]
        bckg   = df[ (df.target==0)&(df.AK4_CSVv2>=0) ]
        backg  = bckg.sample(frac=1)[0:signal.shape[0]]    # equal statistics (& shuffle first!)

        # re-combine into dataframe and shuffle
        self.df = pd.concat( [backg.sample(frac=fraction),signal.sample(frac=fraction)] ).sample(frac=1)

        df = None
        self.metadata = {'metadata':file.get('metadata'),      # names of samples, target values, etc.
                         'offsets':[-1.*i for i in scaler.mean_],
                         'scales':[1./i for i in scaler.scale_]}

        return


    def diagnostics(self,pre=False,post=False):
        """Diagnostic tests of the NN"""
        self.msg_svc.INFO("DL : Diagnostics")

        # Diagnostics before the training
        if pre:
            self.msg_svc.INFO("DL : -- pre-training plots")
            self.plotter.features()                        # compare features for different targets
            self.plotter.correlation()                     # correlations of features
            self.plotter.separation()                      # seprations between features

        # post training/testing
        if post:
            self.msg_svc.INFO("DL : -- post-training plots")
            self.plotter.ROC(self.fpr,self.tpr)  # ROC curve for signal vs background
            self.plotter.history(self.histories) # history metrics as a function of epoch

        return


## THE END ##

