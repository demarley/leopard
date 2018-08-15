
"""
Created:        11 November  2016
Last Updated:   15 February  2018

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

Expandable: Do 'testing' phase later than training phase
            Diagnostics post-training phase
            Different model (PyTorch)
"""
import json
import util
import datetime

import uproot
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential,model_from_json,load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_curve, auc
from deepLearningPlotter import DeepLearningPlotter


# fix random seed for reproducibility
seed = 2018
np.random.seed(seed)


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

        ## NN architecture & parameters -- set by config file
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
        self.activations   = ['elu']         # https://keras.io/activations/
        self.kfold_splits  = 2
        self.nHiddenLayers = 1
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
    def runTraining(self):
        """Train NN model"""
        self.load_hep_data()
        self.build_model()

        # hard-coded :/
        target_names  = ["multijet","QB","W"]
        target_values = [0,1,2]
        self.plotter.initialize(self.df,target_names,target_values)

        if self.runDiagnostics:
            self.diagnostics(preTraining=True)     # save plots of the features and model architecture

        self.train_model()

        self.msg_svc.INFO(" SAVE MODEL")
        self.save_model(self.useLWTNN)

        if self.runDiagnostics:
            self.diagnostics(postTraining=True)    # save plots of the performance in training/testing

        return


    def runInference(self,data=None):
        """
        Run inference of the NN model
        User responsible for diagnostics if not doing training: 
        -> save all predictions (& labels) using 'self.test_predictions'
           then call individual functions:
              plot_features()   -> compare features of the inputs
              plot_prediction() -> compare output prediction (works for classification)
              plot_ROC()        -> signal vs background efficiency (need self.fpr, self.tpr filled)
        """
        self.load_model(self.useLWTNN)

        if data is None:
            try:
                self.load_hep_data()
                data = self.df[self.features]
            except:
                self.msg_svc.ERROR("DL : runInference() cannot proceed because 'data' is None and cannot load HEP data")
                self.msg_svc.ERROR("DL : Please check your implementation.")
                return -999

        prediction = self.predict(data)

        return prediction


    ## Specific functions to perform training/inference tasks
    def build_model(self):
        """Construct the NN model -- only Keras support for now"""
        self.msg_svc.INFO("DL : Build the neural network model")

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

        return



    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        self.msg_svc.INFO("DL : Train the model!")

        callbacks_list = []
        if self.earlystopping:
            earlystop = EarlyStopping(**self.earlystopping)
            callbacks_list = [earlystop]

        ttbar1 = self.df[ self.df['target']==1 ]
        ttbar2 = self.df[ self.df['target']==2 ]

        qcd    = self.df[ self.df['target']==0 ]
        qcd    = qcd.sample(frac=1)[0:ttbar1.shape[0]]   # equal statistics (shuffle QCD entries first)

        training_df = pd.concat( [qcd,ttbar1,ttbar2] )  # re-combine into dataframe
        training_df = training_df.sample(frac=1)         # shuffle entries

        X = training_df[self.features].values  # self.df[self.features].values
        Y = training_df['target'].values       # self.df['target'].values

        kfold = StratifiedKFold(n_splits=self.kfold_splits, shuffle=True, random_state=seed)
        nsplits = kfold.get_n_splits(X,Y)
        cvpredictions = []                 # compare outputs from each cross-validation

        self.msg_svc.INFO("DL :   Fitting K-Fold cross validation".format(self.kfold_splits))
        for ind,(train,test) in enumerate(kfold.split(X,Y)):
            self.msg_svc.DEBUG("DL :   - Fitting K-Fold {0}".format(ind))

            Y_train = Y[train]
            Y_test  = Y[test]

            # - Adjust shape of true values (matrix for multiple outputs)
            if self.dnn_method=='multi' or self.dnn_method=='regression' and not np.array_equal(Y_train,(Y_train[0],self.output_dim)):
                train_shape = Y_train.shape[0]
                train_total_array = []
                test_shape = Y_test.shape[0]
                test_total_array = []

                for a in range(self.output_dim):
                    dummy_train = np.zeros(train_shape)
                    dummy_train[Y_train==a] = 1
                    train_total_array.append( dummy_train.tolist() )

                    dummy_test = np.zeros(test_shape)
                    dummy_test[Y_test==a] = 1
                    test_total_array.append( dummy_test.tolist() )
                Y_train = np.array(train_total_array).T
                Y_test  = np.array(test_total_array).T


            # -- store test/train data from each k-fold to compare later
            self.test_data['X'].append(X[test])
            self.test_data['Y'].append(Y_test)
            self.train_data['X'].append(X[train])
            self.train_data['Y'].append(Y_train)

            ## Fit the model to training data & save the history
            history = self.model.fit(X[train],Y_train,epochs=self.epochs,\
                                     callbacks=callbacks_list,batch_size=self.batch_size,verbose=self.verbose)
            self.histories.append(history)


            # evaluate the model
            self.msg_svc.DEBUG("DL :     + Evaluate the model: ")
            predictions = self.model.evaluate(X[test], Y_test,verbose=self.verbose,batch_size=self.batch_size)
            cvpredictions.append(predictions[1] * 100)
            self.msg_svc.DEBUG("DL :       {0}: {1:.2f}%".format(self.model.metrics_names[1], predictions[1]*100))

            # Evaluate training sample
            train_predictions = self.predict(X[train])
            self.train_predictions.append( train_predictions )

            # Evaluate test sample
            test_predictions  = self.predict(X[test])
            self.test_predictions.append( test_predictions )

            # Make ROC curve from test sample
            if self.dnn_method=='binary':
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
            self.msg_svc.ERROR("DL : Please check your configuration!")
            return -999.
        return self.model.predict( data )


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


    def diagnostics(self,preTraining=False,postTraining=False):
        """Diagnostic tests of the NN"""

        self.msg_svc.INFO("DL : Diagnostics")

        # Diagnostics before the training
        if preTraining:
            self.msg_svc.INFO("DL : -- pre-training")
            self.plotter.features()                   # compare features for different targets
            self.plotter.feature_correlations()       # correlations of features
            self.plotter.model(self.model,self.model_name) # Keras plot of the model architecture

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
