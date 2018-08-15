#ifndef EVENTSELECTION_H
#define EVENTSELECTION_H

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TRandom3.h"
#include "TEfficiency.h"
#include "TF1.h"
#include "TMath.h"
#include "TRandom.h"
#include "TLorentzVector.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "Analysis/leopard/interface/Event.h"
#include "Analysis/leopard/interface/configuration.h"

class eventSelection{

  public:
    // constructor and destructor
    eventSelection(configuration &cmaConfig, const std::string &level="");
    virtual ~eventSelection();

    // Run once at the start of the job to setup the cuts
    virtual void initialize();
    virtual void initialize(const std::string& cutsfile, const std::string& selection);
    virtual void identifySelection();

    // Run for every tree (before the event loop)
    void setCutflowHistograms(TFile& outputFile);

    // Run for every event (in every systematic) that needs saving
    virtual bool applySelection(const Event &event);
    void fillCutflows(double cutflow_bin);  

    // External access to information in this class
    virtual void getCutNames();
    virtual std::vector<std::string> cutNames() {return m_cutflowNames;}     // Return a vector of the cut names (for labeling bins in cutflow histograms)
    virtual unsigned int numberOfCuts() {return m_numberOfCuts;}             // Return the number of cuts (for binning cutflow histograms)

  protected:

    // struct for holding information on a 'cut'
    //  ideally this could be extended so that cuts are parsed & written by code, not humans!
    struct Cut{
        std::string name;       // name of cut
        std::string comparison; // sign of cut (<,<=,>,>=,==,!=)
        float value;            // float value -- cast to int if needed
    };

    configuration* m_config;

    // cut information
    std::string m_level;
    std::string m_selection;
    std::string m_cutsfile;
    unsigned int m_numberOfCuts;
    std::vector<std::string> m_cutflowNames;
    std::vector<Cut> m_cuts;

    // booleans for each selection
    bool m_training;

    // cutflow histograms
    TH1D* m_cutflow;
    TH1D* m_cutflow_unw;

    float m_nominal_weight;
};

#endif
