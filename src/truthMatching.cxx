/*
Created:        25 January 2018
Last Updated:   28 May     2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Match reconstructed objects to truth objects
*/
#include "Analysis/leopard/interface/truthMatching.h"


truthMatching::truthMatching(configuration &cmaConfig) : 
  m_config(&cmaConfig){}

truthMatching::~truthMatching() {}

void truthMatching::initialize(){
    m_truth_tops.clear();
    m_truth_partons.clear();
    return;
}

void truthMatching::setTruthPartons(const std::vector<Parton> truth_partons){
    /* Set truth partons */
    m_truth_partons = truth_partons;
    return;
}
void truthMatching::setTruthTops(const std::vector<TruthTop> truth_tops){
    /* Set truth tops */
    m_truth_tops = truth_tops;
    return;
}


void truthMatching::matchLeptonToTruthTop(Lepton& lepton){
    /* Match lepton to truth top quark */
    lepton.matchId = -1;

    for (unsigned int t_idx=0, size=m_truth_tops.size(); t_idx<size; t_idx++){
        auto truthtop = m_truth_tops.at(t_idx);
        if (truthtop.isHadronic) continue;

        Parton wdecay0 = m_truth_partons.at( truthtop.Wdecays.at(0) );
        Parton wdecay1 = m_truth_partons.at( truthtop.Wdecays.at(1) );
        Parton tlep = (wdecay0.isElectron || wdecay0.isMuon) ? wdecay0 : wdecay1;  // choose electron or muon, not neutrino

        bool ismatched = parton_match(tlep,lepton);    // default DeltaR = 0.2
        if (ismatched) lepton.matchId = t_idx;
    }

    return;
}

void truthMatching::matchLeptonicTopJet(Jet& jet){
    /* Match b from leptonic top decay to AK4 */
    jet.matchId     = -1;
    jet.containment = 0;         // initialize containment
    jet.truth_partons.clear();

    float match_radius(0.75*jet.radius);   // e.g., DeltaR<0.3 rather than 0.4

    for (unsigned int t_idx=0, size=m_truth_tops.size(); t_idx<size; t_idx++){
        auto truthtop = m_truth_tops.at(t_idx);
        if (truthtop.isHadronic) continue;         // only want leptonically-decaying tops

        Parton bottomQ = m_truth_partons.at( truthtop.bottom );
        parton_match(bottomQ,jet,match_radius);
        cma::DEBUG("TRUTHMATCHING : Jet deltaR bottomQ = "+std::to_string(jet.p4.DeltaR(bottomQ.p4))+"; match ID = "+std::to_string(jet.matchId));

        // if the jet is matched to a truth top, exit
        if (jet.containment!=0) jet.matchId = t_idx;
        cma::DEBUG("TRUTHMATCHING : Jet match ID = "+std::to_string(jet.matchId));
    } // end loop over truth tops

    return;
}

void truthMatching::matchJetToTruthTop(Jet& jet){
    /* Match reconstructed or truth jets to partons
       Currently setup to process partons from top quarks (qqb)
    */
    jet.matchId     = -1;
    jet.containment = 0;         // initialize containment
    jet.truth_partons.clear();

    float match_radius( 0.75*jet.radius );   // DeltaR<0.6 rather than 0.8

    for (unsigned int t_idx=0, size=m_truth_tops.size(); t_idx<size; t_idx++){
        auto truthtop = m_truth_tops.at(t_idx);
        if (!truthtop.isHadronic) continue;         // only want hadronically-decaying tops

        Parton bottomQ = m_truth_partons.at( truthtop.bottom );
        Parton wdecay1 = m_truth_partons.at( truthtop.Wdecays.at(0) );
        Parton wdecay2 = m_truth_partons.at( truthtop.Wdecays.at(1) );

        parton_match(bottomQ,jet,match_radius);
        parton_match(wdecay1,jet,match_radius);
        parton_match(wdecay2,jet,match_radius);

        // if the jet is matched to a truth top, exit
        if (jet.containment!=0){
            cma::DEBUG("TRUTHMATCHING : Jet deltaR bottomQ = "+std::to_string(jet.p4.DeltaR(bottomQ.p4)));
            cma::DEBUG("TRUTHMATCHING : Jet deltaR wdecay1 = "+std::to_string(jet.p4.DeltaR(wdecay1.p4)));
            cma::DEBUG("TRUTHMATCHING : Jet deltaR wdecay2 = "+std::to_string(jet.p4.DeltaR(wdecay2.p4)));

            jet.matchId = t_idx;
            break;
        }
    }

    return;
}


bool truthMatching::parton_match(const Parton& p, Lepton& l, float dR){
    /* Match parton to reconstructed jet
       @param p    truth parton
       @param l    reconstructed lepton
       @param dR   distance measure to use in DeltaR (default = -1)
    */
    bool match = (dR > 0) ? cma::deltaRMatch( p.p4, l.p4, dR ) : cma::deltaRMatch( p.p4, l.p4, 0.2 );
    return match;
}


void truthMatching::parton_match(const Parton& p, Jet& j, float dR){
    /* Match parton to reconstructed jet
       @param p    truth parton
       @param j    reconstructed jet
       @param dR   distance measure to use in DeltaR (default = -1)
    */
    bool match = (dR>0) ? cma::deltaRMatch( p.p4, j.p4, dR ) : cma::deltaRMatch( p.p4, j.p4, j.radius );

    // if matched update parameters of the object
    // check matches -- use map in header to avoid errors misremembering the integer values
    if (match){
        cma::DEBUG("TRUTHMATCH : parton_match() "+std::to_string(p.index)+" with containment "+std::to_string(p.containment));
        j.truth_partons.push_back(p.index);
        j.containment += p.containment;
    }

    return;
}


void truthMatching::matchJetToTruthJet(Jet& jet, const std::vector<Jet>& truth_jets){
    /* Match reco jet to truth jet */
    float truthDR( jet.radius );

    for (const auto& truth_jet : truth_jets){
        float thisDR = truth_jet.p4.DeltaR( jet.p4 );
        if ( thisDR < truthDR ){
            jet.truth_jet = truth_jet.index;
            truthDR = thisDR;
        }
    } // end loop over truth jets

    return;
}

// THE END
