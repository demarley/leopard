#ifndef EVENT_H
#define EVENT_H

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TSystem.h"
#include "TEfficiency.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TParameter.h"
#include "TEnv.h"
#include "TF1.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <set>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Analysis/leopard/interface/physicsObjects.h"
#include "Analysis/leopard/interface/configuration.h"
#include "Analysis/leopard/interface/truthMatching.h"
#include "Analysis/leopard/interface/deepLearning.h"
#include "Analysis/leopard/interface/ttbarReco.h"


// Event Class
class Event {
  public:
    // Constructor
    Event( TTreeReader &myReader, configuration &cmaConfig);
    Event( const Event &obj);

    // Destructor
    virtual ~Event();

    // Execute the event (load information and setup objects)
    virtual void execute(Long64_t entry);
    virtual void updateEntry(Long64_t entry);
    bool isValidRecoEntry() const {return (m_entry > (long long)-1);}

    // Setup physics information
    void initialize_jets();
    void initialize_leptons();
    void initialize_kinematics();
    void initialize_weights();
    void initialize_truth();

    // Clear stuff;
    virtual void finalize();
    virtual void clear();

    // Get physics object(s) information
    MET met() const {return m_met;}
    std::vector<Jet> jets() const {return m_jets;}
    std::vector<Lepton> leptons() const {return m_leptons;}

    // metadata
    long long entry() {return m_entry;}
    virtual std::string treeName() {return m_treeName;}

    // functions to calculate things
    bool customIsolation(Lepton& lep);
    float DNN(){ return m_DNN;}

    void buildTtbar();
    std::vector<Top> ttbar() {return m_ttbar;}
    void deepLearningPrediction(Top& top);

    // Get weights
    virtual float nominal_weight() const {return m_nominal_weight;}

  protected:

    // general information
    configuration *m_config;
    TTreeReader &m_ttree;
    TTreeReader m_truth_tree;
    std::string m_treeName;
    bool m_isMC;
    long long m_entry;
    long long m_truth_entry;
    bool m_DNNtraining;
    bool m_DNNinference;

    // event weight information
    double m_nominal_weight;
    double m_xsection;
    double m_kfactor;
    double m_sumOfWeights;
    double m_LUMI;

    // physics object information
    std::vector<Jet> m_jets;
    std::vector<Jet> m_jets_iso;
    std::vector<Jet> m_ak4candidates;
    std::vector<Lepton> m_leptons;
    MET m_met;

    // truth physics object information
    std::vector<Parton> m_truth_partons;
    std::vector<TruthTop> m_truth_tops;

    // Get truth physics information 
    TruthTop m_truth_top;
    TruthTop m_truth_antitop;

    std::map<std::string, double> m_dnnInputs;   // values for inputs to the DNN
    std::string m_dnnKey;
    float m_DNN;   // DNN score

    bool m_useTruth;

    ttbarReco* m_ttbarRecoTool;            // tool to perform ttbar reconstruction
    deepLearning* m_deepLearningTool;      // tool to perform deep learning
    truthMatching* m_truthMatchingTool;    // tool to perform truth-matching
    std::vector<Top> m_ttbar;              // container for ttbar system


    // TTree variables [all possible ones]
    // *************
    // Truth info
    TTreeReaderValue<float> * m_weight_mc;
    TTreeReaderValue<float> * m_weight_pileup;
    TTreeReaderValue<float> * m_weight_lept_eff;
    TTreeReaderValue<float> * m_weight_pileup_UP;
    TTreeReaderValue<float> * m_weight_pileup_DOWN;

    TTreeReaderValue<std::vector<float>> * m_mc_ht;
    TTreeReaderValue<std::vector<float>> * m_mc_pt;
    TTreeReaderValue<std::vector<float>> * m_mc_eta;
    TTreeReaderValue<std::vector<float>> * m_mc_phi;
    TTreeReaderValue<std::vector<float>> * m_mc_e;
    TTreeReaderValue<std::vector<int>> * m_mc_pdgId;
    TTreeReaderValue<std::vector<int>> * m_mc_status;
    TTreeReaderValue<std::vector<int>> * m_mc_isHadTop;
    TTreeReaderValue<std::vector<int>> * m_mc_parent_idx;
    TTreeReaderValue<std::vector<int>> * m_mc_child0_idx;
    TTreeReaderValue<std::vector<int>> * m_mc_child1_idx;

    // MET
    TTreeReaderValue<float> * m_met_met;
    TTreeReaderValue<float> * m_met_phi;

    // Jet info
    TTreeReaderValue<std::vector<float>> * m_jet_pt;
    TTreeReaderValue<std::vector<float>> * m_jet_eta;
    TTreeReaderValue<std::vector<float>> * m_jet_phi;
    TTreeReaderValue<std::vector<float>> * m_jet_m;
    TTreeReaderValue<std::vector<float>> * m_jet_bdisc;
    TTreeReaderValue<std::vector<float>> * m_jet_deepCSV;
    TTreeReaderValue<std::vector<float>> * m_jet_area;
    TTreeReaderValue<std::vector<float>> * m_jet_uncorrPt;
    TTreeReaderValue<std::vector<float>> * m_jet_uncorrE;
    TTreeReaderValue<std::vector<float>> * m_jet_jerSF;
    TTreeReaderValue<std::vector<float>> * m_jet_jerSF_UP;
    TTreeReaderValue<std::vector<float>> * m_jet_jerSF_DOWN;

    // Leptons
    TTreeReaderValue<std::vector<float>> * m_el_pt;
    TTreeReaderValue<std::vector<float>> * m_el_eta;
    TTreeReaderValue<std::vector<float>> * m_el_phi;
    TTreeReaderValue<std::vector<float>> * m_el_e;
    TTreeReaderValue<std::vector<float>> * m_el_charge;
    TTreeReaderValue<std::vector<float>> * m_el_iso;
    TTreeReaderValue<std::vector<unsigned int>> * m_el_id_loose;
    TTreeReaderValue<std::vector<unsigned int>> * m_el_id_medium;
    TTreeReaderValue<std::vector<unsigned int>> * m_el_id_tight;
    TTreeReaderValue<std::vector<unsigned int>> * m_el_id_loose_noIso;
    TTreeReaderValue<std::vector<unsigned int>> * m_el_id_medium_noIso;
    TTreeReaderValue<std::vector<unsigned int>> * m_el_id_tight_noIso;

    TTreeReaderValue<std::vector<float>> * m_mu_pt;
    TTreeReaderValue<std::vector<float>> * m_mu_eta;
    TTreeReaderValue<std::vector<float>> * m_mu_phi;
    TTreeReaderValue<std::vector<float>> * m_mu_e;
    TTreeReaderValue<std::vector<float>> * m_mu_charge;
    TTreeReaderValue<std::vector<float>> * m_mu_iso;
    TTreeReaderValue<std::vector<unsigned int>> * m_mu_id_loose;
    TTreeReaderValue<std::vector<unsigned int>> * m_mu_id_medium;
    TTreeReaderValue<std::vector<unsigned int>> * m_mu_id_tight;

};

#endif
