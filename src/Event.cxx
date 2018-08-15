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
    // AK4
    m_jet_pt  = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4pt");
    m_jet_eta = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4eta");
    m_jet_phi = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4phi");
    m_jet_m   = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4mass");
    m_jet_bdisc    = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4bDisc");
    m_jet_deepCSV  = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4deepCSV");
    m_jet_area     = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4area");
    m_jet_uncorrPt = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4uncorrPt");
    m_jet_uncorrE  = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4uncorrE");
    m_jet_jerSF    = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4jerSF");
    m_jet_jerSF_UP = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4jerSF_UP");
    m_jet_jerSF_DOWN = new TTreeReaderValue<std::vector<float>>(m_ttree,"AK4jerSF_DOWN");

    // Leptons
    m_el_pt  = new TTreeReaderValue<std::vector<float>>(m_ttree,"ELpt");
    m_el_eta = new TTreeReaderValue<std::vector<float>>(m_ttree,"ELeta");
    m_el_phi = new TTreeReaderValue<std::vector<float>>(m_ttree,"ELphi");
    m_el_e   = new TTreeReaderValue<std::vector<float>>(m_ttree,"ELenergy");
    m_el_charge = new TTreeReaderValue<std::vector<float>>(m_ttree,"ELcharge");
    m_el_id_loose  = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"ELlooseID");
    m_el_id_medium = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"ELmediumID");
    m_el_id_tight  = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"ELtightID");
    m_el_id_loose_noIso  = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"ELlooseIDnoIso");
    m_el_id_medium_noIso = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"ELmediumIDnoIso");
    m_el_id_tight_noIso  = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"ELtightIDnoIso");

    m_mu_pt  = new TTreeReaderValue<std::vector<float>>(m_ttree,"MUpt");
    m_mu_eta = new TTreeReaderValue<std::vector<float>>(m_ttree,"MUeta");
    m_mu_phi = new TTreeReaderValue<std::vector<float>>(m_ttree,"MUphi");
    m_mu_e   = new TTreeReaderValue<std::vector<float>>(m_ttree,"MUenergy");
    m_mu_charge = new TTreeReaderValue<std::vector<float>>(m_ttree,"MUcharge");
    m_mu_iso = new TTreeReaderValue<std::vector<float>>(m_ttree,"MUcorrIso");
    m_mu_id_loose  = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"MUlooseID");
    m_mu_id_medium = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"MUmediumID");
    m_mu_id_tight  = new TTreeReaderValue<std::vector<unsigned int>>(m_ttree,"MUtightID");

    // MET
    m_met_met  = new TTreeReaderValue<float>(m_ttree,"METpt");
    m_met_phi  = new TTreeReaderValue<float>(m_ttree,"METphi");


    // set some event weights and access necessary branches
    m_xsection       = 1.0;
    m_kfactor        = 1.0;
    m_sumOfWeights   = 1.0;
    m_LUMI           = m_config->LUMI();

    Sample ss = m_config->sample();

    // MC information
    m_useTruth = (m_config->useTruth());
    if (m_isMC){
      //m_weight_mc    = 1;//new TTreeReaderValue<float>(m_ttree,"evt_Gen_Weight");
      m_xsection     = ss.XSection;
      m_kfactor      = ss.KFactor;        // most likely =1
      m_sumOfWeights = ss.sumOfWeights;

      // TRUTH 
      if (m_config->isTtbar()){
        m_mc_pt  = new TTreeReaderValue<std::vector<float>>(m_ttree,"GENpt");
        m_mc_eta = new TTreeReaderValue<std::vector<float>>(m_ttree,"GENeta");
        m_mc_phi = new TTreeReaderValue<std::vector<float>>(m_ttree,"GENphi");
        m_mc_e   = new TTreeReaderValue<std::vector<float>>(m_ttree,"GENenergy");
        m_mc_pdgId  = new TTreeReaderValue<std::vector<int>>(m_ttree,"GENid");
        m_mc_status = new TTreeReaderValue<std::vector<int>>(m_ttree,"GENstatus");
        m_mc_parent_idx = new TTreeReaderValue<std::vector<int>>(m_ttree,"GENparent_idx");
        m_mc_child0_idx = new TTreeReaderValue<std::vector<int>>(m_ttree,"GENchild0_idx");
        m_mc_child1_idx = new TTreeReaderValue<std::vector<int>>(m_ttree,"GENchild1_idx");
        m_mc_isHadTop = new TTreeReaderValue<std::vector<int>>(m_ttree,"GENisHadTop");
      }
    }

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

    m_leptons.clear();
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

    // Leptons
    initialize_leptons();
    cma::DEBUG("EVENT : Setup lepton ");

    // MET
    initialize_kinematics();
    cma::DEBUG("EVENT : Setup kinematics" );

    // Ttbar Reconstruction
    if (m_leptons.size()>0)
        m_ttbarRecoTool->execute(m_jets,m_leptons.at(0),m_met);
    else{
        // no lepton -- event will fail the selection anyway
        Lepton dummy;
        m_ttbarRecoTool->execute(m_jets,dummy,m_met);
    }
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
    m_truth_tops.clear();

    if (!m_config->isTtbar()) return;   // don't need this for MC other than ttbar

    // only care about this for ttbar
    unsigned int nPartons( (*m_mc_pt)->size() );
    cma::DEBUG("EVENT : N Partons = "+std::to_string(nPartons));

    // Collect truth top information into one value
    unsigned int t_idx(0);  // keeping track of tops in m_truth_tops

    // loop over truth partons
    unsigned int p_idx(0);
    for (unsigned int i=0; i<nPartons; i++){

        Parton parton;
        parton.p4.SetPtEtaPhiE((*m_mc_pt)->at(i),(*m_mc_eta)->at(i),(*m_mc_phi)->at(i),(*m_mc_e)->at(i));

        int status = (*m_mc_status)->at(i);
        int pdgId  = (*m_mc_pdgId)->at(i);
        unsigned int abs_pdgId = std::abs(pdgId);

        parton.pdgId  = pdgId;
        parton.status = status;

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

        parton.index      = p_idx;                    // index in vector of truth_partons
        parton.top_index  = -1;                       // index in truth_tops vector
        parton.containment = NONE;                    // value for containment calculation

        parton.parent_idx = (*m_mc_parent_idx)->at(i);
        parton.child0_idx = (*m_mc_child0_idx)->at(i);
        parton.child1_idx = (*m_mc_child1_idx)->at(i);

        // skip replicated top/W in truth record
        if (parton.isTop && parton.status<60) continue;
        if (parton.isW && (parton.child0_idx<0 || parton.child1_idx<0)) continue;

        // build truth top structs
        // in truth parton record, the top should arrive before its children
        TruthTop top;

        if (parton.isTop){
            cma::DEBUG("EVENT : is top ");
            top.Wdecays.clear();    // for storing W daughters
            top.daughters.clear();  // for storing non-W/bottom daughters

            top.Top       = parton.index;
            top.isTop     = (pdgId>0);
            top.isAntiTop = (pdgId<0);
            top.isHadronic = (*m_mc_isHadTop)->at(p_idx);
            top.isLeptonic = !(*m_mc_isHadTop)->at(p_idx);
            parton.top_index   = t_idx;
            parton.containment = FULL;                            // 'FULL' because it is the top
            if (parton.pdgId<0) parton.containment *= -1;         // negative value for anti-tops

            m_truth_tops.push_back(top);   // store tops now, add information from children in future iterations
            t_idx++;
        }
        else if (!parton.isTop && parton.parent_idx>0) {
            int parent_pdgid  = (*m_mc_pdgId)->at(parton.parent_idx);
            int parent_status = (*m_mc_status)->at(parton.parent_idx);
            cma::DEBUG("EVENT : it's not a top, it's a "+std::to_string(pdgId)
                       +"; parent idx = "+std::to_string(parton.parent_idx)
                       +"; parent pdgid = "+std::to_string(parent_pdgid));

            // check if W is decaying to itself
            if (std::abs(parent_pdgid) == 24 && parent_pdgid == parton.pdgId) {
                bool selfdecay(true);
                int parton_pdgId = parton.pdgId;
                int parton_parent_idx = parton.parent_idx;
                while (selfdecay){
                    if (std::abs(parent_pdgid)==24 && parent_pdgid==parton_pdgId) {
                        int gparent_idx = (*m_mc_parent_idx)->at(parton_parent_idx);
                        // reset parton
                        parton_pdgId  = parent_pdgid;
                        parton_parent_idx = gparent_idx;
                        // reset parent
                        parent_pdgid  = (*m_mc_pdgId)->at(gparent_idx);
                        parent_status = (*m_mc_status)->at(gparent_idx);
                    }
                    else selfdecay=false;
                }
            } // end check for W self-decaying
            else if (parent_pdgid==parton.pdgId) continue;    // other particles self-decaying, just skip

            // get the parent from the list of partons
            Parton parent;
            int top_index(-1);
            unsigned int ptind(0);
            for (const auto& t : m_truth_partons){
                if (t.pdgId==parent_pdgid && t.status==parent_status) {
                    parent    = t;
                    top_index = t.top_index;
                    break;
                }
                ptind++;
            }
            if (top_index<0) continue;    // weird element in truth record, just skip it
            parton.top_index = top_index;

            cma::DEBUG("EVENT : Top index = "+std::to_string(top_index));
            cma::DEBUG("EVENT : - isW?    = "+std::to_string(parent.isW));

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
            else if (parent.isW){
                top = m_truth_tops.at(top_index);
                top.Wdecays.push_back(parton.index);
                top.isHadronic = (parton.isQuark);
                top.isLeptonic = (parton.isLepton);

                parton.containment = QONLY;
                if (top.isAntiTop) parton.containment*=-1;

                m_truth_tops[top_index] = top;      // update entry
            }
        } // end else if not top

        // store for later access
        m_truth_partons.push_back( parton );
        p_idx++;
    } // end loop over truth partons

    m_truthMatchingTool->setTruthPartons(m_truth_partons);
    m_truthMatchingTool->setTruthTops(m_truth_tops);

    return;
}


void Event::initialize_jets(){
    /* Setup struct of jets (small-r) and relevant information */
    unsigned int nJets = (*m_jet_pt)->size();
    m_jets.clear();
    m_jets_iso.clear();       // jet collection for lepton 2D isolation
    m_ak4candidates.clear();  // AK4s to use for training

    unsigned int idx(0);
    unsigned int idx_iso(0);
    for (unsigned int i=0; i<nJets; i++){
        Jet jet;
        jet.p4.SetPtEtaPhiM( (*m_jet_pt)->at(i),(*m_jet_eta)->at(i),(*m_jet_phi)->at(i),(*m_jet_m)->at(i));

        bool isGoodIso( jet.p4.Pt()>15 && std::abs(jet.p4.Eta())<2.4);
        bool isGood(jet.p4.Pt()>50 && std::abs(jet.p4.Eta())<2.4);

        if (!isGood && !isGoodIso) continue;

        jet.isGood = isGood;

        jet.bdisc    = (*m_jet_bdisc)->at(i);
        jet.deepCSV  = (*m_jet_deepCSV)->at(i);
        jet.area     = (*m_jet_area)->at(i);
        jet.uncorrE  = (*m_jet_uncorrE)->at(i);
        jet.uncorrPt = (*m_jet_uncorrPt)->at(i);
        jet.jerSF    = (*m_jet_jerSF)->at(i);
        jet.jerSF_UP = (*m_jet_jerSF_UP)->at(i);
        jet.jerSF_DOWN = (*m_jet_jerSF_DOWN)->at(i);

        jet.index = idx;

        if (isGood){
            m_jets.push_back(jet);

            // truth matching
            if (m_useTruth){
                cma::DEBUG("EVENT : Truth match AK4 jets");
                m_truthMatchingTool->matchLeptonicTopJet(jet);
                if (jet.containment!=0) m_ak4candidates.push_back(jet);
            }

            idx++;
        }
        if (isGoodIso){
            m_jets_iso.push_back(jet);    // used for 2D isolation
            idx_iso++;
        }
    }

    return;
}


void Event::initialize_leptons(){
    /* Setup struct of lepton and relevant information */
    m_leptons.clear();

    // Muons
    unsigned int nMuons = (*m_mu_pt)->size();
    for (unsigned int i=0; i<nMuons; i++){
        Lepton mu;
        mu.p4.SetPtEtaPhiE( (*m_mu_pt)->at(i),(*m_mu_eta)->at(i),(*m_mu_phi)->at(i),(*m_mu_e)->at(i));

        // truth matching
        if (m_useTruth){
            cma::DEBUG("EVENT : Truth match muon");
            m_truthMatchingTool->matchLeptonToTruthTop(mu);
            if (mu.matchId<0) continue;
        }

        bool isMedium = (*m_mu_id_medium)->at(i);
        bool isTight  = (*m_mu_id_tight)->at(i);
        bool iso      = customIsolation(mu);     // 2D isolation cut between leptons & AK4

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

        // truth matching
        if (m_useTruth){
            cma::DEBUG("EVENT : Truth match electron");
            m_truthMatchingTool->matchLeptonToTruthTop(el);
            if (el.matchId<0) continue;
        }

        bool isTightNoIso = (*m_el_id_tight_noIso)->at(i);
        bool iso = customIsolation(el);                     // 2D isolation cut between leptons & AK4

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

    cma::DEBUG("EVENT : Number of leptons = "+std::to_string(m_leptons.size()));

    return;
}


void Event::initialize_kinematics(){
    /* Kinematics from the event (MET) */
    m_met.p4.SetPtEtaPhiM(**m_met_met,0.,**m_met_phi,0.);

    // transverse mass of the W (only relevant for 1-lepton)
    float mtw(0.0);
    if (m_leptons.size()>0){
        Lepton lep = m_leptons.at(0);
        float dphi = m_met.p4.Phi() - lep.p4.Phi();
        mtw = sqrt( 2 * lep.p4.Pt() * m_met.p4.Pt() * (1-cos(dphi)) );
    }
    m_met.mtw = mtw;

    return;
}

void Event::initialize_weights(){
    /* Event weights */
    m_nominal_weight = 1.0;

    if (m_isMC){
        m_nominal_weight  = 1.0; //(**m_weight_pileup) * (**m_weight_mc);
        m_nominal_weight *= (m_xsection) * (m_kfactor) * m_LUMI / (m_sumOfWeights);
    }

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


bool Event::customIsolation( Lepton& lep ){
    /* 2D isolation cut for leptons 
       - Check that the lepton and nearest AK4 jet satisfies
         DeltaR() < 0.4 || pTrel>25
    */
    bool pass(false);
    //int min_index(-1);                    // index of AK4 closest to lep
    float drmin(100.0);                   // min distance between lep and AK4s
    float ptrel(0.0);                     // pTrel between lepton and AK4s

    if (m_jets_iso.size()<1) return false;    // no AK4 -- event will fail anyway

    for (const auto& jet : m_jets_iso){
        float dr = lep.p4.DeltaR( jet.p4 );
        if (dr < drmin) {
            drmin = dr;
            ptrel = cma::ptrel( lep.p4,jet.p4 );
            //min_index = jet.index;
        }
    }

    lep.drmin = drmin;
    lep.ptrel = ptrel;

    if (drmin > 0.4 || ptrel > 30) pass = true;

    return pass;
}


// -- clean-up
void Event::finalize(){
    /* Delete variables used to access information from TTree */
    cma::DEBUG("EVENT : Finalize() ");

    cma::DEBUG("EVENT : Finalize -- Clear Jets");
    delete m_jet_pt;
    delete m_jet_eta;
    delete m_jet_phi;
    delete m_jet_m;
    delete m_jet_bdisc;
    delete m_jet_deepCSV;
    delete m_jet_area;
    delete m_jet_uncorrPt;
    delete m_jet_uncorrE;

    cma::DEBUG("EVENT : Finalize -- Clear leptons");
    delete m_el_pt;
    delete m_el_eta;
    delete m_el_phi;
    delete m_el_e;
    delete m_el_charge;
    delete m_el_id_loose;
    delete m_el_id_medium;
    delete m_el_id_tight;
    delete m_el_id_loose_noIso;
    delete m_el_id_medium_noIso;
    delete m_el_id_tight_noIso;
    delete m_mu_pt;
    delete m_mu_eta;
    delete m_mu_phi;
    delete m_mu_e;
    delete m_mu_charge;
    delete m_mu_iso;
    delete m_mu_id_loose;
    delete m_mu_id_medium;
    delete m_mu_id_tight;

    if (m_isMC){
      cma::DEBUG("EVENT : Finalize -- Clear MC");
      if (m_config->isTtbar()){
        delete m_mc_pt;
        delete m_mc_eta;
        delete m_mc_phi;
        delete m_mc_e;
        delete m_mc_pdgId;
        delete m_mc_status;
        delete m_mc_isHadTop;
      }
    } // end isMC

    return;
}

// THE END
