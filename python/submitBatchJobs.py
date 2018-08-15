"""
Created:        21 September 2016
Last Updated:    3 March     2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Script for using the BatchSubmission() class to submit batch jobs
This is setup for running at the LPC.
If you are running in a different environment, you will need
to modify this script accordingly (mostly concerning the output storage).
This is also designed to submit one batch job at a time.

To run:
$ python python/submitBatchJobs.py batchConfig.txt
where <batchConfig.txt> contains configuration options
"""
import os
import sys
import util
import time
import commands
from batch.batchSubmission import BatchSubmission


# setup config
config = sys.argv[1]
cfg    = util.read_config(config)

# reset boolean values
cfg['test']    = util.str2bool( cfg['test'] )
cfg['submit']  = util.str2bool( cfg['submit'] )
cfg['verbose'] = util.str2bool( cfg['verbose'] )

# set 'global' options
date      = time.strftime("%d%b%Y") if cfg['date']=='today' else cfg['date']
cmaConfig = util.read_config( cfg['config'] )                             # get the selection from the cmaConfig file


# loop over datasets to process, e.g., ttbar or qcd
batch = BatchSubmission()

## -- set arguments for the batch jobs -- ##
batch.username   = cfg['username']
batch.executable = cfg['executable']       # executable to run for the program
batch.test       = cfg['test']             # just submit one job if testing
batch.submit     = cfg['submit']           # do/don't submit jobs, just testing code
batch.verbose    = cfg['verbose']          # lots of print statements
batch.config     = cfg['config']           # configuration file to use
batch.file       = cfg['files']            # individual root files to process
batch.batch_subdir = cfg['subdir']         # sub-directory for storing batch scripts

## Setup output
eos_path       = cfg['eos_path']+"/"+date  # '/store/user/demarley/'+date; separate jobs by date to minimize over-writing
batch.eos_path = eos_path
batch.eos_tarball_path = cfg['eos_tarball_path']


## -- mimic the code in bin/run.cxx and bin/runML.cxx for setting output directory
try:
    customDirectory = cmaConfig["customDirectory"]
except KeyError:
    customDirectory = ''
if customDirectory and not customDirectory.startswith("_"):
    customDirectory = "_{0}".format(customDirectory)

local_output_path = "{0}{1}".format(cmaConfig["selection"].replace(",","-"),customDirectory)
batch.local_output_path = local_output_path          # where to store output

eos_path_full = eos_path+"/"+local_output_path       # files saved in a directory named after the selection
                                                     # as defined in run.cxx/runML.cxx -- update if changed

# create eos directory, if it doesn't exist
os_err = commands.getoutput("eos root://cmseos.fnal.gov/ mkdir -p {0}".format(eos_path_full))
if os_err:
    print "RUNBATCH :: INFO : Attemp to make directory {0}".format(eos_path_full)
    print "RUNBATCH :: INFO : Message = {0}".format(os_err)
else:
    print "RUNBATCH :: INFO : Created directory {0}".format(eos_path_full)


# define the directory to write the output
if cfg['output_dir']=='eos':
    batch.output_dir = 'root://cmseos.fnal.gov/'+eos_path
else:
    batch.output_dir = cfg['output_dir']


## Submit
batch.execute()


## THE END ##
