#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "TROOT.h"
#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "Analysis/leopard/interface/tools.h"


class configuration {
  public:
    // Default - so root can load based on a name;
    configuration( const std::string &configFile );
    //configuration( const configuration& );
    configuration& operator=( const configuration& rhs );

    // Default - so we can clean up;
    virtual ~configuration();

    // Run once at the start of the job;
    virtual void initialize();
    std::string getConfigOption( std::string item );

    // Print configuration
    virtual void print();

    virtual bool isMC();             // must call "inspectFile(file)" or "isMC(file)" first!
    virtual bool isMC( TFile& file );
    bool isTtbar(){ return m_isTtbar;}

    // object declarations
    virtual bool useTruth(){ return m_useTruth;}

    // functions about the TTree
    void setTreename(std::string treeName);
    std::string treename(){ return m_treename;}

    // functions about the file
    virtual void inspectFile( TFile& file );
    std::vector<std::string> filesToProcess(){ return m_filesToProcess;}
    void setFilename(std::string fileName);
    std::string filename(){ return m_filename;}
    std::string primaryDataset() {return m_primaryDataset;}
    unsigned int NTotalEvents() {return m_NTotalEvents;}

    // return some values from config file
    std::string verboseLevel(){ return m_verboseLevel;}
    std::string selection(){ return m_selection;}
    std::string cutsfile(){ return m_cutsfile;}
    std::string outputFilePath(){ return m_outputFilePath;}
    std::string customFileEnding(){ return m_customFileEnding;}
    std::string configFileName(){ return m_configFile;}
    std::string getAbsolutePath(){ return m_cma_absPath;}
    int nEventsToProcess(){ return m_nEventsToProcess;}
    unsigned long long firstEvent(){ return m_firstEvent;}
    bool makeNewFile(){ return m_makeNewFile;}
    bool makeHistograms(){ return m_makeHistograms;}

    // information for event weights
    std::string metadataFile(){ return m_metadataFile;}
    std::map<std::string,Sample> mapOfSamples(){return m_mapOfSamples;}
    Sample sample(){return m_mapOfSamples.at(m_primaryDataset);}
    float LUMI(){ return m_lumi;}

    // DNN
    std::string dnnFile(){ return m_dnnFile;}
    bool DNNtraining(){ return m_DNNtraining;}
    bool DNNinference(){ return m_DNNinference;}
    std::string dnnKey(){ return m_dnnKey;}   // key for lwtnn

  protected:

    void check_btag_WP(const std::string &wkpt);

    std::map<std::string,std::string> m_map_config;
    const std::string m_configFile;

    bool m_isMC;
    bool m_isQCD;
    bool m_isTtbar;
    bool m_useTruth;
    bool m_fileInspected;

    // return some values from config file
    std::string m_selection;
    std::string m_cutsfile;
    std::string m_treename;
    std::string m_filename;
    std::string m_primaryDataset;
    unsigned int m_NTotalEvents;
    std::string m_verboseLevel;
    int m_nEventsToProcess;
    unsigned long long m_firstEvent;
    std::string m_outputFilePath;
    std::string m_customFileEnding;
    bool m_makeNewFile;
    bool m_makeHistograms;
    std::string m_cma_absPath;
    std::string m_metadataFile;
    bool m_DNNtraining;
    bool m_DNNinference;
    std::string m_dnnFile;
    std::string m_dnnKey;

    float m_lumi = 35900;
    std::vector<std::string> m_filesToProcess;
    std::map<std::string,Sample> m_mapOfSamples;  // map of Sample structs

    // Samples primary dataset names
    std::vector<std::string> m_ttbarFiles = {
      "TTToSemilepton_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8",
      "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
      "TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
      "TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
      "TTJets_HT-1200to2500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
      "TTJets_HT-600to800_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
      "TTJets_HT-2500toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
      "TTJets_HT-800to1200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8"}; // possible Ttbar files

    std::map<std::string,std::string> m_defaultConfigs = {
             {"useTruth",              "false"},
             {"makeNewFile",           "true"},
             {"makeHistograms",        "true"},
             {"NEvents",               "-1"},
             {"firstEvent",            "0"},
             {"selection",             "none"},
             {"output_path",           "./"},
             {"customFileEnding",      ""},
             {"cutsfile",              "config/cuts_none.txt"},
             {"inputfile",             "config/inputfiles.txt"},
             {"treename",              "stopTreeMaker/AUX"},
             {"metadataFile",          "config/sampleMetaData.txt"},
             {"verboseLevel",          "INFO"},
             {"DNNtraining",           "false"},
             {"DNNinference",          "false"},
             {"dnnFile",               "config/keras_ttbar_DNN.json"},
             {"dnnKey",                "dnn"} };
};

#endif
