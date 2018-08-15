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
    m_ttree->Branch( "AK4_deepCSVb",  &m_AK4_deepCSVb,  "AK4_deepCSVb/F ");
    m_ttree->Branch( "AK4_deepCSVbb", &m_AK4_deepCSVbb, "AK4_deepCSVbb/F ");
    m_ttree->Branch( "AK4_deepCSVc",  &m_AK4_deepCSVc,  "AK4_deepCSVc/F ");
    m_ttree->Branch( "AK4_deepCSVcc", &m_AK4_deepCSVcc, "AK4_deepCSVcc/F ");
    m_ttree->Branch( "AK4_deepCSVl",  &m_AK4_deepCSVl,  "AK4_deepCSVl/F ");
    m_ttree->Branch( "AK4_deepFlavorb",   &m_AK4_deepFlavorb,   "AK4_deepFlavorb/F ");
    m_ttree->Branch( "AK4_deepFlavorbb",  &m_AK4_deepFlavorbb,  "AK4_deepFlavorbb/F ");
    m_ttree->Branch( "AK4_deepFlavorc",   &m_AK4_deepFlavorc,   "AK4_deepFlavorc/F ");
    m_ttree->Branch( "AK4_deepFlavoruds", &m_AK4_deepFlavoruds, "AK4_deepFlavoruds/F ");
    m_ttree->Branch( "AK4_deepFlavorg",   &m_AK4_deepFlavorg,   "AK4_deepFlavorg/F ");
    m_ttree->Branch( "AK4_deepFlavorlepb",&m_AK4_deepFlavorlepb,"AK4_deepFlavorlepb/F ");
    m_ttree->Branch( "AK4_charge", &m_AK4_charge, "AK4_charge/F" );


    /**** Metadata ****/
    // which sample has which target value
    // many ROOT files will be merged together to do the training
    m_metadataTree->Branch( "name",    &m_name );
    m_metadataTree->Branch( "target",  &m_target_value,  "target/I" );
    m_metadataTree->Branch( "nEvents", &m_nEvents,       "nEvents/I" );

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
    m_AK4_deepCSVb  = features.at("AK4_deepCSVb");
    m_AK4_deepCSVbb = features.at("AK4_deepCSVbb");
    m_AK4_deepCSVc  = features.at("AK4_deepCSVc");
    m_AK4_deepCSVcc = features.at("AK4_deepCSVcc");
    m_AK4_deepCSVl  = features.at("AK4_deepCSVl");

    cma::DEBUG("MINITREE : Save event2b ");
    m_AK4_deepFlavorb  = features.at("AK4_deepFlavorb");
    m_AK4_deepFlavorbb = features.at("AK4_deepFlavorbb");
    m_AK4_deepFlavorc  = features.at("AK4_deepFlavorc");
    m_AK4_deepFlavorg  = features.at("AK4_deepFlavorg");
    m_AK4_deepFlavoruds  = features.at("AK4_deepFlavoruds");
    m_AK4_deepFlavorlepb = features.at("AK4_deepFlavorlepb");

    m_AK4_charge = features.at("AK4_charge");

    /**** Fill the tree ****/
    cma::DEBUG("MINITREE : Fill the tree");
    m_ttree->Fill();

    return;
}


void miniTree::finalize(){
    /* Finalize the class -- fill in the metadata (only need to do this once!) */
    m_name    = m_config->primaryDataset();
    m_nEvents = m_config->NTotalEvents();
    m_target_value = (m_config->isQCD()) ? 0 : -1;    // multiple classes for signal, choose '-1'

    cma::DEBUG("MINITREE : Fill the metadata tree");
    m_metadataTree->Fill();
}

// THE END
