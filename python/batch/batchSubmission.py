"""
Created:        21 September 2016
Last Updated:    3 March     2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Base class for submitting batch jobs.
Setup for condor system at the LPC
-- Class can be extended for other batch systems
"""
import os
import sys
import commands
from collections import OrderedDict
import Analysis.leopard.util as util
import batchScripts as bs


class BatchSubmission(object):
    """Class for submitting batch jobs in different environments"""
    def __init__(self):
        """Initialize some variables for the class"""
        self.username   = 'dmarley'
        self.executable = 'run'
        self.baseDir    = os.environ['PWD']+'/'
        self.test       = False     # just submit one job if testing
        self.submit     = True      # don't submit jobs, just testing code
        self.config     = 'config/cmaConfig.txt' # CyMiniAna configuration file
        self.file       = 'share/miniSL_ntuples.txt'

        self.cmssw_base = os.environ['CMSSW_BASE']            # path to CMSSW directory
        self.cmsRelease = os.environ['CMSSW_VERSION']         # 'CMSSW_8_0_28_patch1'
        self.eos_path   = '/store/user/lpctop/ttbarAC/'
        self.eos_tarball_path  = '/store/user/lpctop/ttbarAC/'
        self.local_output_path = ''

        self.batch_subdir = ''
        self.cfg_filename = "batch/batchConfig_{0}.txt" # CyMiniAna config file per job
        self.filename     = ''

        self.verbose_level = "INFO"    # print output to screen
        self.vb = None


    def initialize(self):
        """Setup a few variables based on class attributes"""
        self.unique_id_base = [self.tmp_ntuple]
        self.unique_id_name = "_".join(self.unique_id_base)
        self.unique_id_path = "/".join(self.unique_id_base)
        if self.batch_subdir:
            self.unique_id_batch_path = "batch/{0}/{1}".format(self.batch_subdir,self.unique_id_path)
        else:
            self.unique_id_batch_path = "batch/{0}".format(self.unique_id_path)


        self.vb = util.VERBOSE()
        self.vb.level = self.verbose_level
        self.vb.initialize()

        # Setup options for CyMiniAna
        self.config_options = OrderedDict()
        config = open(self.config,'r').readlines()
        for i in config:
            j = i.rstrip('\n').split(' ')
            key,item = j
            self.config_options[key] = item

        return



    def writeConfiguration(self):
        """Write the CyMiniAnaAC configuration file"""
        config_text = ""
        for key in self.config_options:
            value = self.config_options[key]

            if key == "inputfile":
                config_text += "inputfile {0}\n".format(self.filename)
            else:
                config_text += "{0} {1}\n".format(key,value)

        cfg_file = open(self.cfg_filename,"w")
        cfg_file.write(config_text)
        cfg_file.close()

        return


    def writeNtuple2File(self,minisl_ntuple):
        """
           Write the name of the root file to a text file.
           - CyMiniAnaAC uses a text file to read the root files that it processes.
        """
        tmp_file = open(self.filename,"w")
        tmp_file.write(minisl_ntuple)
        tmp_file.close()

        return


    def writeBatchScript(self):
        """Write the batch submission script(s)"""
        self.vb.DEBUG("BATCH SUBMISSION : Write batch scripts ")

        # Write the shell script executed in condor job
        condorExecFileName = "{0}/run_condor.sh".format(self.unique_id_batch_path)
        condor_executable  = bs.condor_bash_template() % {'outputDir':self.output_dir,
            'executable':self.executable,'eos_path':self.eos_path,'cmsRelease':self.cmsRelease,
            'cfg_filename':self.cfg_filename,'executable':self.executable,'eos_path_to_tarball':self.eos_tarball_path,
            'username':self.username,'local_output_path':self.local_output_path,'unique_id_batch_path':self.unique_id_batch_path}

        # Write the condor submission script
        template = bs.condor_script_template() % {'baseDir':self.baseDir,
            'condorExec':condorExecFileName,'unique_id_batch_path':self.unique_id_batch_path}

        # need executable that sets up the environment properly
        condorExecFile = open(condorExecFileName,'w')
        condorExecFile.write(condor_executable)
        commands.getoutput("chmod +x {0}".format(condorExecFileName))

        batchFile = open(self.batchFileName,'w')
        batchFile.write(template)

        return



    def execute(self):
        """Execute the script"""
        # -- Print the configuration
        if self.verbose_level == "DEBUG": self.Print()

        # -- Make some directories, if needed
        if not os.path.exists('batch'): os.makedirs('batch')

        # -- Submit jobs
        if not self.file.startswith("/"): self.file = self.baseDir+"/"+self.file

        miniSL_ntuples = open(self.file,'r').readlines() # files to run over

        for l,line in enumerate(miniSL_ntuples):
            minisl_ntuple = line.split()[0]

            # -- simplify the filename for reference later
            self.tmp_ntuple = minisl_ntuple.split('/')[-1].replace('.root','')


            # -- Setup the unique ID for making unique files/configurations/etc.
            self.initialize()
            if not os.path.exists(self.unique_id_batch_path):
                os.makedirs(self.unique_id_batch_path)

            ## Put the single root file into a text file
            self.filename = "{0}/listOfFiles.txt".format(self.unique_id_batch_path)
            self.writeNtuple2File(minisl_ntuple)

            self.vb.INFO("BATCH SUBMISSION : {0}".format(minisl_ntuple))
            self.vb.INFO("BATCH SUBMISSION : >> {0}".format(self.tmp_ntuple))

            self.submit_job()

        return


    def submit_job(self):
        """Submit the job"""
        # -- Write the CyMiniAnaAC configuration file
        self.cfg_filename = "{0}/cmaConfig.txt".format(self.unique_id_batch_path)
        self.writeConfiguration()
        self.vb.INFO("BATCH SUBMISSION : Config filename = {0}".format(self.cfg_filename))

        # -- Write the script executed by the batch system
        self.batchFileName = "{0}/run_batch.condor".format(self.unique_id_batch_path)
        self.writeBatchScript()
        self.vb.INFO("BATCH SUBMISSION : Batch filename  = {0}".format(self.batchFileName))


        # only submit the jobs if you want to (True by default)
        if self.submit:
            batch_cmd_str = "condor_submit {0}".format(self.batchFileName)
            batch_command = commands.getoutput( batch_cmd_str )

            self.vb.INFO("BATCH SUBMISSION : {0}".format(batch_cmd_str))
            self.vb.INFO("BATCH SUBMISSION : {0}".format(batch_command))


        # if testing job submission, submit only one and exit
        if self.test:
            self.vb.INFO("BATCH SUBMISSION : Test complete. ")
            self.vb.INFO("BATCH SUBMISSION : Exiting. ")
            sys.exit(1)

        return



    def Print(self):
        """Print the configuration of arguments for batch submission"""
        print ""
        print " BATCH SUBMISSION : Configuration "
        print " -------------------------------- "
        print ""

        attributes = ['username','executable','config','baseDir',
        'mode','test','submit','verbose_level']

        for attr in attributes:
            print " %-*s : %s" % (16,attr,getattr(self,attr))

        print ""
        print " -------------------------------- "
        print ""

        return



## THE END
