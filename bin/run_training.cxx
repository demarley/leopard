/*
Created:        14 August 2018
Last Updated:   14 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Basic steering macro for running leopard
 - Make flat ntuples for machine learning
 - Make histograms to compare features
*/
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TH1.h"
#include "TSystem.h"
#include "TMath.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <map>
#include <fstream>
#include <string>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Analysis/leopard/interface/configuration.h"
#include "Analysis/leopard/interface/Event.h"
#include "Analysis/leopard/interface/eventSelection.h"
#include "Analysis/leopard/interface/miniTree.h"
#include "Analysis/leopard/interface/tools.h"
#include "Analysis/leopard/interface/histogrammer.h"


int main(int argc, char** argv) {
    /* Steering macro for leopard */
    if (argc < 2) {
        cma::HELP();
        return -1;
    }

    unsigned long long maxEntriesToRun(0);     // maximum number of entries in TTree
    unsigned int numberOfEventsToRun(0);       // number of events to run
    bool passEvent(false);                     // event passed selection

    // configuration
    configuration config(argv[1]);             // configuration file
    config.initialize();

    int nEvents = config.nEventsToProcess();                      // requested number of events to run
    std::string outpathBase = config.outputFilePath();            // directory for output files
    unsigned long long firstEvent      = config.firstEvent();     // first event to begin running over
    std::vector<std::string> filenames = config.filesToProcess(); // list of files to process
    std::string selection(config.selection());                    // selection to apply
    std::string treename(config.treename());                      // name of TTree

    std::string customFileEnding( config.customFileEnding() );
    if (customFileEnding.length()>0  && customFileEnding.substr(0,1).compare("_")!=0){
        customFileEnding = "_"+customFileEnding; // add '_' to beginning of string, if needed
    }

    // event selection
    eventSelection evtSel( config );
    evtSel.initialize();


    // --------------- //
    // -- File loop -- //
    // --------------- //
    unsigned int numberOfFiles(filenames.size());
    unsigned int currentFileNumber(0);

    cma::INFO("TRAINING : *** Starting file loop *** ");
    for (const auto& filename : filenames) {

        ++currentFileNumber;
        cma::INFO("TRAINING :   Opening "+filename+"   ("+std::to_string(currentFileNumber)+"/"+std::to_string(numberOfFiles)+")");

        auto file = TFile::Open(filename.c_str());
        if (!file || file->IsZombie()){
            cma::WARNING("TRAINING :  -- File: "+filename);
            cma::WARNING("TRAINING :     does not exist or it is a Zombie. ");
            cma::WARNING("TRAINING :     Continuing to next file. ");
            continue;
        }

        // check that the ttree exists in this file before proceeding
        std::vector<std::string> fileKeys;
        cma::getListOfKeys(file,fileKeys);
        if (std::find(fileKeys.begin(), fileKeys.end(), treename) == fileKeys.end()){
            cma::INFO("TRAINING : TTree "+treename+" is not present in this file, continuing to next TTree");
            continue;
        }


        config.setFilename( filename );   // Use the filename to determine primary dataset and information about the sample
        config.inspectFile( *file );      // Determine information about the input file (metadata)
        Sample s = config.sample();       // load the Sample struct (xsection,kfactor,etc)


        // -- Output file -- //
        // For each event selection, make a new output directory
        std::string outpath = outpathBase+"/"+selection+customFileEnding;

        std::string outputFilename     = cma::setupOutputFile(outpath,filename);
        std::string fullOutputFilename = outpath+"/"+outputFilename+".root";
        std::unique_ptr<TFile> outputFile(TFile::Open( fullOutputFilename.c_str(), "RECREATE"));
        cma::INFO("TRAINING :   >> Saving to "+fullOutputFilename);


        // -- Load TTree to loop over
        cma::INFO("TRAINING :      TTree "+treename);
        TTreeReader myReader(treename.c_str(), file);

        // -- Setup cutflow histograms
        evtSel.setCutflowHistograms(*outputFile);

        // -- Initialize histograms
        histogrammer histMaker(config,"ML");
        histMaker.initialize( *outputFile );

        // -- Make new Tree in Root file
        miniTree miniTTree(config);
        miniTTree.initialize( *outputFile );

        // -- Number of Entries to Process -- //
        maxEntriesToRun = myReader.GetEntries(true);
        if (maxEntriesToRun<1) // skip files with no entries
            continue;

        if (nEvents < 0 || ((unsigned int)nEvents+firstEvent) > maxEntriesToRun)
            numberOfEventsToRun = maxEntriesToRun - firstEvent;
        else
            numberOfEventsToRun = nEvents;

        // ---------------- //
        // -- Event Loop -- //
        // ---------------- //
        Long64_t imod = 1;                     // print to the terminal
        Event event = Event(myReader, config);

        Long64_t eventCounter = 0;    // counting the events processed
        Long64_t entry = firstEvent;  // start at a different event!
        while (myReader.Next()) {

            if (eventCounter+1 > numberOfEventsToRun){
                cma::INFO("TRAINING : Processed the desired number of events: "+std::to_string(eventCounter)+"/"+std::to_string(numberOfEventsToRun));
                break;
            }

            if (entry%imod==0){
                cma::INFO("TRAINING :       Processing event "+std::to_string(entry) );
                if(imod<2e4) imod *=10;
            }

            // -- Build Event -- //
            cma::DEBUG("TRAINING : Execute event");
            event.execute(entry);
            // now we have event object that has the event-level objects in it
            // pass this to the selection tools

            // -- Event Selection -- //
            cma::DEBUG("TRAINING : Apply event selection");
            passEvent = evtSel.applySelection(event);

            std::vector<Top> tops = event.ttbar();
            if (passEvent && tops.size()>0){
                cma::DEBUG("TRAINING : Passed selection, now save information");

                // For ML, we are training on semi-boosted top quarks (AK8+AK4)
                // Only save features of the AK8+AK4 system to the output ntuple/histograms
                std::map<std::string,double> features2save;
                features2save["xsection"] = s.XSection;
                features2save["kfactor"]  = s.KFactor;
                features2save["sumOfWeights"] = s.sumOfWeights;
                features2save["nominal_weight"] = event.nominal_weight();

                for (const auto& top : tops){

                    for (const auto& x : top.dnn) features2save[x.first] = x.second;

                    miniTTree.saveEvent(features2save);
                    histMaker.fill(features2save);
                }
            }

            // iterate the entry and number of events processed
            ++entry;
            ++eventCounter;
        } // end event loop

        event.finalize();
        miniTTree.finalize();

        // put overflow/underflow content into the first and last bins
        histMaker.overUnderFlow();

        cma::INFO("TRAINING :   END Running  "+filename);
        cma::INFO("TRAINING :   >> Output at "+fullOutputFilename);

        outputFile->Write();
        outputFile->Close();

        // -- Clean-up stuff
        delete file;          // free up some memory 
        file = ((TFile *)0);  // (no errors for too many root files open)
    } // end file loop

    cma::INFO("TRAINING : *** End of file loop *** ");
    cma::INFO("TRAINING : Program finished. ");
}

// THE END
