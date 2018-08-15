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
  m_config(&cmaConfig){}

ttbarReco::~ttbarReco() {}


void ttbarReco::execute(const std::vector<Jet>& jets, const Lepton& lepton, const MET& met){
    /* Build top quarks system */
    m_ttbar.clear();

    // Combine information into one struct
    for (const auto& jet : jets){
        Top top_cand;  // reconstructed top candidates
        top_cand.jet = jet.index;
        top_cand.lepton = lepton;
        top_cand.met    = met;

        // signal if the AK4 comes from the same top quark as the lepton else background
        unsigned int target(0);
        if (jet.matchId==lepton.matchId)
            target = 1;

        top_cand.target = target;
        m_ttbar.push_back( top_cand );
    }

    cma::DEBUG("TTBARRECO : Ttbar built ");

    return;
}

// THE END //
