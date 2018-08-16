#ifndef TRUTHMATCHING_H
#define TRUTHMATCHING_H

#include "TROOT.h"
#include "TFile.h"
#include "TSystem.h"

#include <string>
#include <map>
#include <vector>

#include "Analysis/leopard/interface/tools.h"
#include "Analysis/leopard/interface/configuration.h"
#include "Analysis/leopard/interface/physicsObjects.h"

class truthMatching {
  public:

    // Default
    truthMatching(configuration &cmaConfig);

    // Default - so we can clean up;
    virtual ~truthMatching();
    void initialize();
    void setTruthPartons(const std::vector<Parton> truth_partons);
    void setTruthTops(const std::vector<TruthTop> truth_tops);

    void matchLeptonToTruthTop(Lepton& lepton) const;
    void matchLeptonicTopJet(Jet& jet) const;
    void matchJetToTruthTop(Jet& jet) const;
    void matchJetToTruthJet(Jet& jet, const std::vector<Jet>& truth_jets) const;

    void parton_match(const Parton& p, Jet& j, float dR=-1.0) const;
    bool parton_match(const Parton& p, Lepton& l, float dR=-1.0) const;

  protected:

    configuration *m_config;

    std::vector<TruthTop> m_truth_tops;
    std::vector<Parton> m_truth_partons;
};

#endif
