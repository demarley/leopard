/*
Created:        14 August 2018
Last Updated:   14 August  2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Event class
 Contains all the objects (& structs) with event information
*/
#include "Analysis/leopard/interface/Event.h"

// constructor
Event::Event( TTreeReader &myReader, configuration &cmaConfig ) :
  m_config(&cmaConfig),
  m_ttree(myReader),
  m_treeName("SetMe"),
  m_DNN(-999.),
  m_useTruth(false){
    m_isMC     = m_config->isMC();
    m_treeName = m_ttree.GetTree()->GetName();              // for systematics
    m_DNNtraining  = m_config->DNNtraining();               // load DNN inputs for training
    m_DNNinference = m_config->DNNinference();              // load DNN inputs for inference

    // ** LOAD BRANCHES FROM TTREE ** //
    // Event Info -- Filters and Triggers
    m_eeBadScFilter   = new TTreeReaderValue<int>(m_ttree,"eeBadScFilter");
    m_BadPFMuonFilter = new TTreeReaderValue<unsigned int>(m_ttree,"BadPFMuonFilter");
    m_noBadMuonsFilter = new TTreeReaderValue<int>(m_ttree,"noBadMuonsFilter");
    m_badMuonsFilter   = new TTreeReaderValue<int>(m_ttree,"badMuonsFilter");
    m_duplicateMuonsFilter = new TTreeReaderValue<int>(m_ttree,"duplicateMuonsFilter");
    m_HBHENoiseFilter = new TTreeReaderValue<unsigned int>(m_ttree,"HBHENoiseFilter");
    m_goodVerticesFilter = new TTreeReaderValue<int>(m_ttree,"goodVerticesFilter");
    m_HBHEIsoNoiseFilter = new TTreeReaderValue<unsigned int>(m_ttree,"HBHEIsoNoiseFilter");
    m_globalTightHalo2016Filter = new TTreeReaderValue<int>(m_ttree,"globalTightHalo2016Filter");
    m_BadChargedCandidateFilter = new TTreeReaderValue<unsigned int>(m_ttree,"BadChargedCandidateFilter");
    m_EcalDeadCellTriggerPrimitiveFilter = new TTreeReaderValue<int>(m_ttree,"EcalDeadCellTriggerPrimitiveFilter");

    m_PassTrigger  = new TTreeReaderValue<std::vector<int>>(m_ttree,"PassTrigger");
    m_TriggerNames = new TTreeReaderValue<std::vector<std::string>>(m_ttree,"TriggerNames");

    // AK4
    m_NJetsISR = new TTreeReaderValue<int>(m_ttree,"NJetsISR");
    m_ak4LVec  = new TTreeReaderValue<std::vector<TLorentzVector>>(m_ttree,"jetsLVec");
    m_ak4Flavor = new TTreeReaderValue<std::vector<int>>(m_ttree,"recoJetsFlavor");
    m_ak4Charge = new TTreeReaderValue<std::vector<double>>(m_ttree,"recoJetsCharge_0");
    m_ak4looseJetID = new TTreeReaderValue<unsigned int>(m_ttree,"looseJetID");
    m_ak4tightJetID = new TTreeReaderValue<unsigned int>(m_ttree,"tightJetID");
    m_ak4tightlepvetoJetID = new TTreeReaderValue<unsigned int>(m_ttree,"tightlepvetoJetID");
    m_ak4deepCSV_b  = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepCSVb");
    m_ak4deepCSV_bb = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepCSVbb");
    m_ak4deepCSV_c  = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepCSVc");
    m_ak4deepCSV_cc = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepCSVcc");
    m_ak4deepCSV_l  = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepCSVl");
    m_ak4deepFlavor_b    = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepFlavorb");
    m_ak4deepFlavor_bb   = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepFlavorbb");
    m_ak4deepFlavor_lepb = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepFlavorlepb");
    m_ak4deepFlavor_c    = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepFlavorc");
    m_ak4deepFlavor_uds  = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepFlavoruds");
    m_ak4deepFlavor_g    = new TTreeReaderValue<std::vector<double>>(m_ttree,"DeepFlavorg");
    m_ak4qgLikelihood = new TTreeReaderValue<std::vector<double>>(m_ttree,"qgLikelihood");
    m_ak4qgPtD   = new TTreeReaderValue<std::vector<double>>(m_ttree,"qgPtD");
    m_ak4qgAxis1 = new TTreeReaderValue<std::vector<double>>(m_ttree,"qgAxis1");
    m_ak4qgAxis2 = new TTreeReaderValue<std::vector<double>>(m_ttree,"qgAxis2");
    m_ak4qgMult  = new TTreeReaderValue<std::vector<int>>(m_ttree,"qgMult");

    // TRUTH
    m_useTruth = (m_config->useTruth());
    if (m_isMC){
        m_selPDGid   = new TTreeReaderValue<std::vector<int>>(m_ttree,"selPDGid");
        m_genMatched = new TTreeReaderValue<std::vector<double>>(m_ttree,"genMatched");
        m_genDecayLVec   = new TTreeReaderValue<std::vector<TLorentzVector>>(m_ttree,"genDecayLVec");
        m_genDecayIdxVec = new TTreeReaderValue<std::vector<int>>(m_ttree,"genDecayIdxVec");
        m_genDecayPdgIdVec  = new TTreeReaderValue<std::vector<int>>(m_ttree,"genDecayPdgIdVec");
        m_genDecayMomIdxVec = new TTreeReaderValue<std::vector<int>>(m_ttree,"genDecayMomIdxVec");
        m_genDecayMomRefVec = new TTreeReaderValue<std::vector<int>>(m_ttree,"genDecayMomRefVec");

        m_stored_weight = new TTreeReaderValue<double>(m_ttree,"stored_weight");
    } // end isMC

    // Truth matching tool
    m_truthMatchingTool = new truthMatching(cmaConfig);
    m_truthMatchingTool->initialize();

    m_ttbarRecoTool    = new ttbarReco(cmaConfig);
    m_deepLearningTool = new deepLearning(cmaConfig);

    // DNN material
} // end constructor

Event::~Event() {}

void Event::clear(){
    /* Clear many of the vectors/maps for each event -- SAFETY PRECAUTION */
    m_truth_partons.clear();
    m_truth_tops.clear();

    m_lepton = {};
    m_jets.clear();

    m_dnnInputs.clear();

    return;
}


void Event::updateEntry(Long64_t entry){
    /* Update the entry -> update all TTree variables */
    cma::DEBUG("EVENT : Update Entry "+std::to_string(entry) );

    m_entry = entry;

    // make sure the entry exists/is valid
    if (isValidRecoEntry())
        m_ttree.SetEntry(m_entry);
    else
        cma::ERROR("EVENT : Invalid Reco entry "+std::to_string(m_entry)+"!");

    cma::DEBUG("EVENT : Set entry for updating ");

    return;
}


void Event::execute(Long64_t entry){
    /* Get the values from the event */
    cma::DEBUG("EVENT : Execute event " );

    // Load data from root tree for this event
    updateEntry(entry);

    // Reset many event-level values
    clear();

    // Get the event weights (for cutflows/histograms)
    initialize_weights();
    cma::DEBUG("EVENT : Setup weights ");

    // Truth Information (before other physics objects, to do truth-matching)
    if (m_useTruth && m_isMC){
        initialize_truth();
        cma::DEBUG("EVENT : Setup truth information ");
    }

    // Jets
    initialize_jets();
    cma::DEBUG("EVENT : Setup small-R jets ");

    // Lepton
    initialize_lepton();
    cma::DEBUG("EVENT : Setup lepton ");

    // Ttbar Reconstruction
    m_ttbarRecoTool->execute(m_jets,m_lepton);
    m_ttbar = m_ttbarRecoTool->tops();

    // DNN prediction for each Top object
    for (auto& top : m_ttbar)
        deepLearningPrediction(top);

    cma::DEBUG("EVENT : Setup Event ");

    return;
}


void Event::initialize_truth(){
    /* Setup struct of truth information */
    m_truth_partons.clear();
    unsigned int nPartons( (*m_genDecayLVec)->size() );

    // Collect truth top information into one value
    unsigned int t_idx(0);  // keeping track of tops in m_truth_tops
    m_truth_tops.clear();

    // loop over truth partons
    for (unsigned int i=0; i<nPartons; i++){
        Parton parton;
        parton.p4 = (*m_genDecayLVec)->at(i);

        int pdgId = (*m_genDecayPdgIdVec)->at(i);
        unsigned int abs_pdgId = std::abs(pdgId);

        parton.pdgId = pdgId;

        // simple booleans for type
        parton.isTop = ( abs_pdgId==6 );
        parton.isW   = ( abs_pdgId==24 );
        parton.isLepton = ( abs_pdgId>=11 && abs_pdgId<=16 );
        parton.isQuark  = ( abs_pdgId<7 );

        if (parton.isLepton){
            parton.isTau  = ( abs_pdgId==15 );
            parton.isMuon = ( abs_pdgId==13 );
            parton.isElectron = ( abs_pdgId==11 );
            parton.isNeutrino = ( abs_pdgId==12 || abs_pdgId==14 || abs_pdgId==16 );
        }
        else if (parton.isQuark){
            parton.isLight  = ( abs_pdgId<5 );
            parton.isBottom = ( abs_pdgId==5 );
        }

        parton.index      = i;                              // index in vector of truth_partons
        parton.decayIdx   = (*m_genDecayIdxVec)->at(i);     // index in full truth record of parton
        parton.parent_ref = (*m_genDecayMomRefVec)->at(i);  // index in truth vector of parent
        parton.parent_idx = (*m_genDecayMomIdxVec)->at(i);  // index in full truth record of parent
        parton.top_index  = -1;                             // index in truth_tops vector
        parton.containment = 0;                             // value for determining matching

        // build truth top structs
        // in truth parton record, the top should arrive before its children
        TruthTop top;

        if (parton.isTop){
            top.Wdecays.clear();    // for storing W daughters
            top.daughters.clear();  // for storing non-W/bottom daughters

            top.Top       = parton.index;
            top.isTop     = (pdgId>0);
            top.isAntiTop = (pdgId<0);
            top.isHadronic = false;     // initialize
            top.isLeptonic = false;     // initialize
            parton.top_index = t_idx;

            m_truth_tops.push_back(top);   // store tops now, add information from children in the next iterations over partons
            t_idx++;
        }
        else if (parton.parent_ref>0){
            // check the parent! (ignore parent_ref<=0 -- doesn't exist)
            Parton parent = m_truth_partons.at(parton.parent_ref);

            // check if grandparent exists
            int top_index(-1);            // to refer to (grand)parent top quark in m_truth_tops
            if(parent.parent_ref>0) {
                Parton gparent = m_truth_partons.at(parent.parent_ref);  // parent of parent
                if (gparent.isTop) top_index = gparent.top_index;
            }

            // Parent is Top (W or b)
            if (parent.isTop){
                top = m_truth_tops.at(parent.top_index);
                if (parton.isW) top.W = parton.index;
                else if (parton.isBottom) {
                    top.bottom = parton.index;
                    parton.containment = BONLY;
                    if (top.isAntiTop) parton.containment*=-1;
                }
                else top.daughters.push_back( parton.index );        // non-W/bottom daughter
                m_truth_tops[parent.top_index] = top;                // update entry
            }
            // Parent is W
            else if (parent.isW && top_index>=0){
                top = m_truth_tops.at(top_index);
                top.Wdecays.push_back(parton.index);
                top.isHadronic = (parton.isQuark);
                top.isLeptonic = (parton.isLepton);

                parton.containment = QONLY;
                if (top.isAntiTop) parton.containment*=-1;

                m_truth_tops[top_index] = top;      // update entry
            }
        } // end else if

        // store for later access
        m_truth_partons.push_back( parton );
    } // end loop over truth partons

    m_truthMatchingTool->setTruthPartons(m_truth_partons);
    m_truthMatchingTool->setTruthTops(m_truth_tops);

    return;
}


void Event::initialize_jets(){
    /* Setup struct of jets (small-r) and relevant information */
    m_jets.clear();  // don't know a priori the number of jets that pass kinematics
    m_ak4candidates.clear();

    unsigned int j_idx(0); // counting jets that pass kinematic cuts
    for (unsigned int i=0,size=(*m_ak4LVec)->size(); i<size; i++){
        Jet jet;
        jet.p4 = (*m_ak4LVec)->at(i);

        // kinematic cuts
        if (jet.p4.Pt() < 30 || std::abs( jet.p4.Eta() > 2.4 ) ) continue;

        // Other properties
        jet.charge = (*m_ak4Charge)->at(i);
        jet.true_flavor = (m_isMC) ? (*m_ak4Flavor)->at(i) : -1;
        jet.index  = j_idx;
        jet.radius = 0.4;
        jet.containment = 0;   // initialize in case this is data

        jet.deepCSVb  = (*m_ak4deepCSV_b)->at(i);
        jet.deepCSVbb = (*m_ak4deepCSV_bb)->at(i);
        jet.deepCSVc  = (*m_ak4deepCSV_c)->at(i);
        jet.deepCSVcc = (*m_ak4deepCSV_cc)->at(i);
        jet.deepCSVl  = (*m_ak4deepCSV_l)->at(i);

        jet.deepFlavorb  = (*m_ak4deepFlavor_b)->at(i);
        jet.deepFlavorbb = (*m_ak4deepFlavor_bb)->at(i);
        jet.deepFlavorc  = (*m_ak4deepFlavor_c)->at(i);
        jet.deepFlavorg  = (*m_ak4deepFlavor_g)->at(i);
        jet.deepFlavoruds  = (*m_ak4deepFlavor_uds)->at(i);
        jet.deepFlavorlepb = (*m_ak4deepFlavor_lepb)->at(i);

        // truth matching
        if (m_useTruth){
            cma::DEBUG("EVENT : Truth match AK4 jets");

            // parton
            m_truthMatchingTool->matchJetToTruthTop(jet);
            if (jet.containment!=0) m_ak4candidates.push_back(jet.index);
        }

        m_jets.push_back(jet);
        j_idx++;
    }

    return;
}


void Event::initialize_leptons(){
    /* Setup struct of lepton and relevant information */
    m_leptons.clear();
    m_electrons.clear();
    m_muons.clear();

    // Muons
    unsigned int nMuons = (*m_mu_pt)->size();
    for (unsigned int i=0; i<nMuons; i++){
        Lepton mu;
        mu.p4.SetPtEtaPhiE( (*m_mu_pt)->at(i),(*m_mu_eta)->at(i),(*m_mu_phi)->at(i),(*m_mu_e)->at(i));
        bool isMedium   = (*m_mu_id_medium)->at(i);
        bool isTight    = (*m_mu_id_tight)->at(i);

        bool iso = customIsolation(mu);    // 2D isolation cut between leptons & AK4 (need AK4 initialized first!)

        bool isGood(mu.p4.Pt()>60 && std::abs(mu.p4.Eta())<2.4 && isMedium && iso);
        if (!isGood) continue;

        mu.charge = (*m_mu_charge)->at(i);
        mu.loose  = (*m_mu_id_loose)->at(i);
        mu.medium = isMedium; 
        mu.tight  = isTight; 
        mu.iso    = (*m_mu_iso)->at(i);
        mu.isGood = isGood;

        mu.isMuon = true;
        mu.isElectron = false;

        m_leptons.push_back(mu);
    }

    // Electrons
    unsigned int nElectrons = (*m_el_pt)->size();
    for (unsigned int i=0; i<nElectrons; i++){
        Lepton el;
        el.p4.SetPtEtaPhiE( (*m_el_pt)->at(i),(*m_el_eta)->at(i),(*m_el_phi)->at(i),(*m_el_e)->at(i));
        bool isTightNoIso = (*m_el_id_tight_noIso)->at(i);

        bool iso = customIsolation(el);    // 2D isolation cut between leptons & AK4 (need AK4 initialized first!)

        bool isGood(el.p4.Pt()>60 && std::abs(el.p4.Eta())<2.4 && isTightNoIso && iso);
        if (!isGood) continue;

        el.charge = (*m_el_charge)->at(i);
        el.loose  = (*m_el_id_loose)->at(i);
        el.medium = (*m_el_id_medium)->at(i);
        el.tight  = (*m_el_id_tight)->at(i);
        el.loose_noIso  = (*m_el_id_loose_noIso)->at(i);
        el.medium_noIso = (*m_el_id_medium_noIso)->at(i);
        el.tight_noIso  = isTightNoIso;
        el.isGood = isGood;

        el.isMuon = false;
        el.isElectron = true;

        m_leptons.push_back(el);
    }

    return;
}



void Event::initialize_weights(){
    /* Event weights */
    m_nominal_weight = 1.0;
    if (m_isMC) m_nominal_weight = **m_stored_weight; //**m_evtWeight;

    return;
}


void Event::deepLearningPrediction(Top& top){
    /* Return map of deep learning values */
    if (m_DNNtraining){
        cma::DEBUG("EVENT : Calculate DNN features for training ");
        m_deepLearningTool->training(top);
        top.dnn = m_deepLearningTool->features();
    }

    if (m_DNNinference){
        cma::DEBUG("EVENT : Calculate DNN ");
        m_deepLearningTool->inference(top);
        top.dnn = m_deepLearningTool->features();
    }

    return;
}


// -- clean-up
void Event::finalize(){
    /* Delete variables used to access information from TTree */
    cma::DEBUG("EVENT : Finalize() ");
    delete m_PassTrigger;
    delete m_TriggerNames;

    delete m_BadChargedCandidateFilter;
    delete m_BadPFMuonFilter;
    delete m_EcalDeadCellTriggerPrimitiveFilter;
    delete m_HBHEIsoNoiseFilter;
    delete m_HBHENoiseFilter;
    delete m_eeBadScFilter;
    delete m_globalTightHalo2016Filter;
    delete m_goodVerticesFilter;

    cma::DEBUG("EVENT : Finalize -- Clear Jets");

    if (m_isMC){
      cma::DEBUG("EVENT : Finalize -- Clear MC");
      delete m_stored_weight;

      delete m_selPDGid;
      delete m_genDecayIdxVec;
      delete m_genDecayLVec;
      delete m_genDecayMomIdxVec;
      delete m_genDecayMomRefVec;
      delete m_genDecayPdgIdVec;
    } // end isMC

    return;
}

// THE END
