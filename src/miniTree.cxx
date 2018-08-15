/*
Created:        20 February 2018
Last Updated:   20 May      2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Create and fill TTree for ML
*/
#include "Analysis/leopard/interface/miniTree.h"


miniTree::miniTree(configuration &cmaConfig) : 
  m_config(&cmaConfig){}

miniTree::~miniTree() {}


void miniTree::initialize(TFile& outputFile) {
    /*
       Setup the new tree 
       Contains features for the NN
       --  No vector<T> stored in tree: completely flat!
    */
    outputFile.cd();                                     // move to output file
    m_ttree        = new TTree("features", "features");  // Tree contains features for the NN
    m_metadataTree = new TTree("metadata","metadata");   // Tree contains metadata

    /**** Setup new branches here ****/
    // Weights
    m_ttree->Branch( "xsection", &m_xsection, "xsection/F" );
    m_ttree->Branch( "kfactor",  &m_kfactor,  "kfactor/F" );
    m_ttree->Branch( "weight",   &m_weight,   "weight/F" );
    m_ttree->Branch( "sumOfWeights",   &m_sumOfWeights,   "sumOfWeights/F" );
    m_ttree->Branch( "nominal_weight", &m_nominal_weight, "nominal_weight/F" );

    // Features
    m_ttree->Branch( "target", &m_target, "target/I" );  // target value (.e.g, 0 or 1)

    // AK4
    m_ttree->Branch( "AK4_CSVv2",        &m_AK4_CSVv2,        "AK4_CSVv2/F" );
    m_ttree->Branch( "mass_lep_AK4",     &m_mass_lep_AK4,     "mass_lep_AK4/F" );
    m_ttree->Branch( "deltaR_lep_AK4",   &m_deltaR_lep_AK4,   "deltaR_lep_AK4/F" );
    m_ttree->Branch( "ptrel_lep_AK4",    &m_ptrel_lep_AK4,    "ptrel_lep_AK4/F" );
    m_ttree->Branch( "deltaPhi_met_AK4", &m_deltaPhi_met_AK4, "deltaPhi_met_AK4/F" );
    m_ttree->Branch( "deltaPhi_met_lep", &m_deltaPhi_met_lep, "deltaPhi_met_lep/F" );

    /**** Metadata ****/
    m_metadataTree->Branch( "name",    &m_name );
    m_metadataTree->Branch( "nEvents", &m_nEvents, "nEvents/I" );

    return;
} // end initialize



void miniTree::saveEvent(const std::map<std::string,double> features) {
    /* Save the ML features to the ttree! */
    cma::DEBUG("MINITREE : Save event ");

    m_weight   = features.at("weight");
    m_kfactor  = features.at("kfactor");
    m_xsection = features.at("xsection");
    m_sumOfWeights = features.at("sumOfWeights");
    m_nominal_weight = features.at("nominal_weight");

    m_target = features.at("target");

    cma::DEBUG("MINITREE : Save event2a ");
    m_AK4_CSVv2      = features.at("AK4_CSVv2");
    m_mass_lep_AK4   = features.at("mass_lep_AK4");
    m_deltaR_lep_AK4 = features.at("deltaR_lep_AK4");
    m_ptrel_lep_AK4  = features.at("ptrel_lep_AK4");
    m_deltaPhi_met_AK4 = features.at("deltaPhi_met_AK4");
    m_deltaPhi_met_lep = features.at("deltaPhi_met_lep");

    /**** Fill the tree ****/
    cma::DEBUG("MINITREE : Fill the tree");
    m_ttree->Fill();

    return;
}


void miniTree::finalize(){
    /* Finalize the class -- fill in the metadata (only need to do this once!) */
    m_name    = m_config->primaryDataset();
    m_nEvents = m_config->NTotalEvents();

    cma::DEBUG("MINITREE : Fill the metadata tree");
    m_metadataTree->Fill();
}

// THE END
