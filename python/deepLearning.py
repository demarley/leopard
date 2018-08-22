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
import collections

from deepLearningPlotter import DeepLearningPlotter

import uproot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DeepLearning(object):
    """Deep Learning base class"""
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

        print self.activations
        print self.nNodes
        print self.nHiddenLayers

        return


    ## Single functions to run all of the necessary pieces
    def training(self):
        """Train NN model"""
        pass

    def inference(self,data=None):
        """
        Run inference of the NN model
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
        pass

    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        pass

    def load_model(self,from_lwtnn=False):
        """Load existing model to make plots or predictions"""
        pass

    def save_model(self,to_lwtnn=False):
        """Save the model for use later"""
        pass

    def predict(self,data=None):
        """Return the prediction from a test sample"""
        pass


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
        fraction=0.01
        signal = df[ (df.target==1)&(df.AK4_CSVv2>=0) ]
        bckg   = df[ (df.target==0)&(df.AK4_CSVv2>=0) ]
        backg  = bckg.sample(frac=1)[0:signal.shape[0]]    # equal statistics (& shuffle first!)

        # re-combine into dataframe and shuffle
        self.df = pd.concat( [backg.sample(frac=fraction),signal.sample(frac=fraction)] ).sample(frac=1)

        del df
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
