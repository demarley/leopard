/*
Created:        14 August 2018
Last Updated:   29 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Make histograms!
*/
#include "Analysis/leopard/interface/histogrammer.h"


histogrammer::histogrammer( configuration& cmaConfig, std::string name ) :
  m_config(&cmaConfig),
  m_name(name),
  m_isMC(false),
  m_useLargeRJets(false),
  m_useJets(false),
  m_putOverflowInLastBin(true),
  m_putUnderflowInFirstBin(true){
    m_map_histograms1D.clear();
    m_map_histograms2D.clear();

    m_isMC = m_config->isMC();

    if (m_name.length()>0  && m_name.substr(m_name.length()-1,1).compare("_")!=0)
        m_name = m_name+"_"; // add '_' to end of string, if needed
  }

histogrammer::~histogrammer() {}


/**** INITIALIZE HISTOGRAMS ****/

// -- 1D Histograms
void histogrammer::init_hist( const std::string& name, const unsigned int nBins, const double x_min, const double x_max ){
    /* Initialize histogram -- equal bins */
    m_map_histograms1D["h_"+name] = new TH1D(("h_"+name).c_str(), ("h_"+name).c_str(),nBins,x_min,x_max);
    m_map_histograms1D["h_"+name]->Sumw2();

    return;
}
void histogrammer::init_hist( const std::string& name, const unsigned int nBins, const double *xbins ){
    /* Initialize histogram -- variable bins */
    m_map_histograms1D["h_"+name] = new TH1D(("h_"+name).c_str(), ("h_"+name).c_str(),nBins,xbins);
    m_map_histograms1D["h_"+name]->Sumw2();

    return;
}
// -- 2D Histograms
void histogrammer::init_hist( const std::string& name, const unsigned int nBinsX, const double x_min, const double x_max,
                              const unsigned int nBinsY, const double y_min, const double y_max ){
    /* Initialize histogram -- equal bins */
    m_map_histograms2D["h_"+name] = new TH2D(("h_"+name).c_str(), ("h_"+name).c_str(),
                                             nBinsX,x_min,x_max,nBinsY,y_min,y_max);
    m_map_histograms2D["h_"+name]->Sumw2();

    return;
}
void histogrammer::init_hist( const std::string& name, const unsigned int nBinsX, const double *xbins,
                              const unsigned int nBinsY, const double *ybins ){
    /* Initialize histogram -- variable bins */
    m_map_histograms2D["h_"+name] = new TH2D(("h_"+name).c_str(), ("h_"+name).c_str(),
                                             nBinsX,xbins,nBinsY,ybins);
    m_map_histograms2D["h_"+name]->Sumw2();

    return;
}


void histogrammer::initialize( TFile& outputFile, bool doSystWeights ){
    /* Setup some values and book histograms */
    m_doSystWeights = doSystWeights;
    outputFile.cd();

    bookHists( m_name );

    return;
}

void histogrammer::bookHists( std::string name ){
    /* 
      Book histograms -- modify/inherit this function for analysis-specific hists 

      @param name   This is the string used to identify histograms for different systematics/event weights
    */
    m_names.resize(0); // append names to this to keep track of later

    cma::DEBUG("HISTOGRAMMER : Init. histograms: "+name);

    // features for DNN
    for (unsigned int x=0, size=m_targets.size(); x<size; x++){
        std::string target = std::to_string(x);

        // AK4
        init_hist("AK4_deepCSVb_"+target+"_"+m_name,  200, -1,1);
        init_hist("AK4_deepCSVbb_"+target+"_"+m_name, 200, -1,1);
        init_hist("AK4_deepCSVc_"+target+"_"+m_name,  200, -1,1);
        init_hist("AK4_deepCSVcc_"+target+"_"+m_name, 200, -1,1);
        init_hist("AK4_deepCSVl_"+target+"_"+m_name,  200, -1,1);
        init_hist("AK4_deepFlavorb_"+target+"_"+m_name,   200, -1,1);
        init_hist("AK4_deepFlavorbb_"+target+"_"+m_name,  200, -1,1);
        init_hist("AK4_deepFlavorc_"+target+"_"+m_name,   200, -1,1);
        init_hist("AK4_deepFlavoruds_"+target+"_"+m_name, 200, -1,1);
        init_hist("AK4_deepFlavorg_"+target+"_"+m_name,   200, -1,1);
        init_hist("AK4_deepFlavorlepb_"+target+"_"+m_name,200, -1,1);

        init_hist("AK4_charge_"+target+"_"+m_name, 500, -5,5);

        // AK8 + AK4 system
        init_hist("AK8AK4_deltaR_"+target+"_"+m_name, 50,0,5);
        init_hist("AK8AK4_mass_"+target+"_"+m_name,  500,0,1000);
        init_hist("AK8AK4_mass_deltaR_"+target+"_"+m_name,    500,0,1000,50,0,5); // (AK8+AK4).M() vs DeltaR(AK8,AK4)

    return;
}




/**** FILL HISTOGRAMS ****/
void histogrammer::fill( const std::string& name, const double& value, const double& weight ){
    /* TH1D */
    TH1D* this_hist = m_map_histograms1D.at("h_"+name);
    this_hist->Fill(value,weight);
    return;
}

void histogrammer::fill( const std::string& name, 
                         const double& xvalue, const double& yvalue, const double& weight ){
    /* TH2D */
    TH2D* this_hist = m_map_histograms2D.at("h_"+name);
    this_hist->Fill(xvalue,yvalue,weight);
    return;
}

void histogrammer::fill( Event& event ){
    /* Fill histograms -- fill histograms based on treename or systematic weights ("nominal" but different weight)
       This is the function to modify / inherit for analysis-specific purposes
    */
    std::string treeName = event.treeName();
    double event_weight  = event.nominal_weight();

    fill( m_name+treeName, event, event_weight );

    // for different event weights, e.g., systematics,
    // put other calls to fill() here with that change

    return;
}


void histogrammer::fill( const std::string& name, Event& event, double event_weight){
    /* Fill histograms -- just use information from the event and fill histogram
       This is the function to modify / inherit for analysis-specific purposes
    */
    cma::DEBUG("HISTOGRAMMER : Fill histograms: "+name+".");

    // Load some objects from Event
    std::vector<Top> tops = event.ttbar();  // reconstructed ttbar system

    cma::DEBUG("HISTOGRAMMER : event weight = "+std::to_string(event_weight) );

        cma::DEBUG("HISTOGRAMMER : Fill small-R jets");
        fill("n_jets_"+name, jets.size(), event_weight );

        for (const auto& jet : jets){
            fill("jet_pt_"+name,  jet.p4.Pt(),   event_weight);
            fill("jet_eta_"+name, jet.p4.Eta(),  event_weight);
            fill("jet_phi_"+name, jet.p4.Phi(),  event_weight);
            fill("jet_bdisc_"+name, jet.bdisc, event_weight);
        }
    }

    cma::DEBUG("HISTOGRAMMER : End histograms");

    return;
}





/**** OVER/UNDERFLOW ****/

void histogrammer::overUnderFlow(){
    /* Call overflow and underflow functions at once */
    overFlow();
    underFlow();
    return;
}


void histogrammer::overFlow() {
    /* Add overflow to last bin */
    if (!m_putOverflowInLastBin){
        cma::INFO("HISTOGRAMMER : Not putting overflow in last bin(s)");
        return;
    }
    else{
        // loop over 1D histograms
        for (const auto& hist : m_map_histograms1D){
            unsigned int numBins = hist.second->GetNbinsX();
            double overflowContent = hist.second->GetBinContent(numBins + 1);

            hist.second->AddBinContent(numBins,overflowContent); // add overflow to last bin
            hist.second->SetBinContent(numBins+1, 0);            // set overflow to 0
        }
        // loop over 2D histograms
        for (const auto& hist : m_map_histograms2D){
            unsigned int numBinsX = hist.second->GetXaxis()->GetNbins();
            unsigned int numBinsY = hist.second->GetYaxis()->GetNbins();

            // x-axis :: overflow in y
            for (unsigned int xx=1;xx<numBinsX+1;++xx){
                double overflowContent = hist.second->GetBinContent(xx,numBinsY+1);

                int lastBin     = hist.second->GetBin(xx,numBinsY);
                int overflowBin = hist.second->GetBin(xx,numBinsY+1);
                hist.second->AddBinContent(lastBin,overflowContent); // add overflow to last bin
                hist.second->SetBinContent(overflowBin,0);           // set overflow to 0
            }
            // y-axis :: overflow in x
            for (unsigned int yy=1;yy<numBinsY;++yy){
                double overflowContent = hist.second->GetBinContent(numBinsX+1,yy);

                int lastBin     = hist.second->GetBin(numBinsX,yy);
                int overflowBin = hist.second->GetBin(numBinsX+1,yy);
                hist.second->AddBinContent(lastBin,overflowContent); // add overflow to last bin
                hist.second->SetBinContent(overflowBin,0);           // set overflow to 0
            }
        } // end 2D histogram overflow
    } // end else put overflow in first bin

    return;
}

void histogrammer::underFlow() {
    /* Add underflow to first bin */
    if (!m_putUnderflowInFirstBin){
        cma::INFO("HISTOGRAMMER : Not putting underflow in first bin(s)");
        return;
    }
    else{
        // loop over 1D histograms
        for (const auto& hist : m_map_histograms1D){
            double underflowContent = hist.second->GetBinContent(0);

            hist.second->AddBinContent(1, underflowContent);  // add underflow to first bin
            hist.second->SetBinContent(0, 0);                 // set underflow to 0
        }
        // loop over 2D histograms
        for (const auto& hist : m_map_histograms2D){
            unsigned int numBinsX = hist.second->GetXaxis()->GetNbins();
            unsigned int numBinsY = hist.second->GetYaxis()->GetNbins();

            // x-axis :: underflow in y
            for (unsigned int xx=1;xx<numBinsX+1;++xx){
                double underflowContent = hist.second->GetBinContent(xx,numBinsY+1);

                int firstBin     = hist.second->GetBin(xx,1);
                int underflowBin = hist.second->GetBin(xx,0);
                hist.second->AddBinContent(firstBin,underflowContent); // add overflow to last bin
                hist.second->SetBinContent(underflowBin,0);            // set overflow to 0
            }
            // y-axis :: underflow in x
            for (unsigned int yy=1;yy<numBinsY;++yy){
                double underflowContent = hist.second->GetBinContent(0,yy);

                int firstBin     = hist.second->GetBin(1,yy);
                int underflowBin = hist.second->GetBin(0,yy);
                hist.second->AddBinContent(firstBin,underflowContent); // add overflow to last bin
                hist.second->SetBinContent(underflowBin,0);           // set overflow to 0
            }
        } // end 2D histogram underflow
    } // end else put underflow in first bin
}

// THE END