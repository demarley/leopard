/*
Created:        19 February 2018
Last Updated:   28 May      2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Tool for performing deep learning tasks
*/
#include "Analysis/leopard/interface/deepLearning.h"


deepLearning::deepLearning( configuration& cmaConfig ) :
  m_config(&cmaConfig),
  m_lwnn(nullptr),
  m_dnnKey(""){

    if (m_config->DNNinference()){
        // Setup lwtnn
        std::ifstream input_cfg = cma::open_file( m_config->dnnFile() );
        lwt::JSONConfig cfg     = lwt::parse_json( input_cfg );
        m_lwnn   = new lwt::LightweightNeuralNetwork(cfg.inputs, cfg.layers, cfg.outputs);
        m_dnnKey = m_config->dnnKey();
    }
  }

deepLearning::~deepLearning() {
    delete m_lwnn;
}


void deepLearning::training(Top& top){
    /* Prepare inputs for training */
    loadFeatures(top);

    return;
}

void deepLearning::inference(Top& top){
    /* Obtain results from LWTNN */
    loadFeatures(top);
    m_discriminant = m_lwnn->compute(m_dnnInputs);
    top.dnn = m_discriminant;

    return;
}


void deepLearning::loadFeatures(const Top& top){
    /* Calculate DNN features */
    m_dnnInputs.clear();

    // feature calculations
    m_dnnInputs["target"] = top.target;

    Jet jet = top.jet;
    MET met = top.met;
    Lepton lep = top.lepton;

    m_dnnInputs["AK4_CSVv2"]      = jet.bdisc;
    m_dnnInputs["mass_lep_AK4"]   = (jet.p4+lep.p4).M();
    m_dnnInputs["deltaR_lep_AK4"] = jet.p4.DeltaR(lep.p4);
    m_dnnInputs["ptrel_lep_AK4"]  = cma::ptrel( jet.p4,lep.p4 );
    m_dnnInputs["deltaPhi_met_AK4"] = jet.p4.DeltaPhi(met.p4);
    m_dnnInputs["deltaPhi_met_lep"] = lep.p4.DeltaPhi(met.p4);

    m_dnnInputs["weight"] = 1. / (lep.p4 + jet.p4).Pt();  // 1/ljet.p4.Pt() or something

    cma::DEBUG("EVENT : Set DNN input values ");

    return;
}

std::map<std::string,double> deepLearning::features(){
    /* return features */
    return m_dnnInputs;
}

std::map<std::string,double> deepLearning::predictions(){
    /* Return the full map to the user */
    return m_discriminant;
}

double deepLearning::prediction(){
    /* Return the score for the default key */
    return m_discriminant.at(m_dnnKey);
}

double deepLearning::prediction(const std::string& key){
    /* Just return the prediction (after execute!) */
    return m_discriminant.at(key);
}

// THE END //
