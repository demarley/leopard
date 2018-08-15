/*
Created:        14 August 2018
Last Updated:   14 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Tool for performing deep learning tasks
*/
#include "Analysis/leopard/interface/ttbarReco.h"


ttbarReco::ttbarReco( configuration& cmaConfig ) :
  m_config(&cmaConfig){
    m_targetMap = m_config->mapOfTargetValues();
  }

ttbarReco::~ttbarReco() {}


void ttbarReco::execute(const std::vector<Jet>& jets, const Lepton& lepton){
    /* Build top quarks system */
    m_ttbar.clear();

    m_jets = jets;
    m_lepton = lepton;


    // Reconstruct ttbar, define containment
    bool isTtbar(m_config->isTtbar());

    Top top_cand;  // reconstructed top candidates
    top_cand.jets.clear();

    for (const auto& jet : ak4candidates){
        top_cand.jets.push_back(jet.index);
        top_cand.target = target;
        m_ttbar.push_back( top_cand );
    } // end loop over AK4

    cma::DEBUG("TTBARRECO : Ttbar built ");

    return;
}

// THE END //
