#ifndef TTBARRECO_H
#define TTBARRECO_H

#include <string>
#include <map>
#include <vector>

#include "Analysis/leopard/interface/tools.h"
#include "Analysis/leopard/interface/configuration.h"
#include "Analysis/leopard/interface/physicsObjects.h"


class ttbarReco {
  public:
    ttbarReco( configuration& cmaConfig );

    ~ttbarReco();

    std::vector<Top> tops(){ return m_ttbar;}
    void execute(const std::vector<Jet>& jets, const Lepton& lepton);

  protected:

    configuration *m_config;

    std::vector<Top> m_ttbar;

    Lepton m_lepton;
    std::vector<Jet> m_jets;
};

#endif
