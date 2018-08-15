"""
Created:         3 March 2018
Last Updated:    3 March 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Templates for scripts to submit batch jobs
"""


def condor_bash_template():
    """Bash script that will run Goldilocks -- designed for condor"""
    bash_template = """#!/bin/tcsh
echo " > Starting CONDOR job at `date` on `hostname`"


## Copy CMSSW tarball from eos, untar it, and remove the tarball
xrdcp -s root://cmseos.fnal.gov/%(eos_path_to_tarball)s/%(cmsRelease)s.tgz .
tar -xf %(cmsRelease)s.tgz
rm %(cmsRelease)s.tgz

## copy transferred data in batch directory (only need 3 files)
if (! -d %(cmsRelease)s/src/Analysis/goldilocks/%(unique_id_batch_path)s/ ) then
    mkdir -p %(cmsRelease)s/src/Analysis/goldilocks/%(unique_id_batch_path)s/
endif

mv cmaConfig.txt %(cmsRelease)s/src/Analysis/goldilocks/%(unique_id_batch_path)s/
mv listOfFiles.txt %(cmsRelease)s/src/Analysis/goldilocks/%(unique_id_batch_path)s/
mv run_condor.sh %(cmsRelease)s/src/Analysis/goldilocks/%(unique_id_batch_path)s/

## Setup CMSSW environment
setenv SCRAM_ARCH slc6_amd64_gcc530
cd %(cmsRelease)s/src/
scramv1 b ProjectRename
eval `scramv1 runtime -csh` # cmsenv is an alias not on the workers
scram b -j8


## Execute goldilocks
cd Analysis/goldilocks

echo " > Run the program "
%(executable)s %(cfg_filename)s


## Move output file to EOS (make new directory if necessary)
echo " > Finishing executing "
echo " > Move output file to EOS"

# -- Only copy files if they exist
set outfiles = `ls %(local_output_path)s/`
if ( "$outfiles" != "" ) then
    echo " Output file exists, copy to EOS "
    xrdcp %(local_output_path)s/*.root %(outputDir)s/%(local_output_path)s
endif


## Cleanup
rm %(local_output_path)s/*.root   # delete output files
cd ${_CONDOR_SCRATCH_DIR}         # delete working directory
rm -rf %(cmsRelease)s

echo " > Ended at `date` on `hostname`"
ls -alh
exit 0
"""

    return bash_template




def condor_script_template():
    """Condor script for submitting batch job"""
    condor_template = """
Universe   = vanilla
Executable = %(condorExec)s

Should_Transfer_Files = YES
WhenToTransferOutput  = ON_EXIT
Transfer_Input_Files  = %(unique_id_batch_path)s/cmaConfig.txt,%(unique_id_batch_path)s/listOfFiles.txt,%(unique_id_batch_path)s/run_condor.sh
notify_user   = ${LOGNAME}@FNAL.GOV
x509userproxy = $ENV(X509_USER_PROXY)

Log        = %(unique_id_batch_path)s/condor_$(Cluster)_$(Process).log
Error	   = %(unique_id_batch_path)s/condor_$(Cluster)_$(Process).error
Output     = %(unique_id_batch_path)s/condor_$(Cluster)_$(Process).out

Queue 1
"""

    return condor_template


## THE END ##
