#ifndef PHYSICSOBJECTS_H
#define PHYSICSOBJECTS_H

/* 
   Physics objects to be used in analyses
   This structure allows the Event class
   and other classes to access these objects
   without circular inclusion (which breaks!)
*/
#include "TLorentzVector.h"
#include <map>
#include <string>

// 'containment' for matching jets to partons from tops
enum tmatch {NONE,BONLY,QONLY,BQ,W,FULL};


// base object (consistent reference to TLorentzVector)
struct CmaBase {
    TLorentzVector p4;
    bool isGood;
};


// Truth information
struct Parton : CmaBase {
    int pdgId;
    int status;
    int index;       // index in vector of truth partons
    int decayIdx;    // index in truth record
    int parent_ref;  // index in truth vector of parent
    int parent_idx;  // index in truth record of parent
    int child0_idx;  // index in truth record of child0
    int child1_idx;  // index in truth record of child1
    int containment;
    int top_index;   // index of parton in the truth_top vector

    // Heavy Object Booleans
    bool isWprime;
    bool isVLQ;

    bool isTop;
    bool isW;
    bool isZ;
    bool isHiggs;
    // Lepton Booleans
    bool isLepton;
    bool isTau;
    bool isElectron;
    bool isMuon;
    bool isNeutrino;
    // Quark Booleans
    bool isQuark;
    bool isBottom;
    bool isLight;
};

struct TruthTop {
    // collect indices in truth_partons vector of top parton info
    bool isTop;
    bool isAntiTop;
    int Top;
    int W;
    int bottom;
    std::vector<int> Wdecays;   // for storing W daughters
    std::vector<int> daughters; // for storing non-W/bottom daughters

    bool isHadronic;  // W decays to quarks
    bool isLeptonic;  // W decays to leptons
};


// Reco information
struct Jet : CmaBase {
    // extra jet attributes
    float bdisc;
    std::map<std::string, bool> isbtagged;
    int true_flavor;
    float radius;
    double charge;
    double rho;      // jet energy density, 1 value per event (attaching to each jet for convenience)
    int index;       // index in vector of jets

    int truth_jet;   // index in vector of truth jets that is closest to this jet
    int containment; // level of containment for partons
    std::vector<int> truth_partons;  // vector containing partons that are truth-matched to jet
    int matchId;    // keep track of jets matched to top or anti-top

    float area;      // area of jet (needed to redo JEC)
    float uncorrPt;  // area of jet (needed to redo JEC)
    float uncorrE;   // area of jet (needed to redo JEC)
    float jerSF;     // JER smearing for MC
    float jerSF_UP;
    float jerSF_DOWN;

    float deepCSV;
    float deepCSVb;
    float deepCSVbb;
    float deepCSVc;
    float deepCSVcc;
    float deepCSVl;

    float deepFlavorb;
    float deepFlavorbb;
    float deepFlavorc;
    float deepFlavoruds;
    float deepFlavorg;
    float deepFlavorlepb;
};

// Extra lepton attributes
struct Lepton : CmaBase{
    // common to electrons and muons
    int charge;
    bool isElectron;
    bool isMuon;
    int index;       // index in vector of leptons
    int matchId;     // record keeping of to which top quark this originated

    float drmin;     // distance to closest AK4
    float ptrel;     // relative pT to closest AK4
    float iso;
    float id;
    int loose;
    int medium;
    int tight;
    int loose_noIso;
    int medium_noIso;
    int tight_noIso;
};


struct MET : CmaBase{
    // extra MET attributes
    float mtw;   // transverse mass of W
};


struct Top {
    // Define a top quark
    TLorentzVector p4;
    unsigned int target;        // for ML training
    std::map<std::string,double> dnn;

    bool isTop;
    bool isAntiTop;

    MET met;
    Lepton lepton;
    Jet jet;
};



// ------------------------ // 
// Struct to contain sample information (processing the input file)

struct Sample {
    std::string primaryDataset;
    float XSection;
    float KFactor;
    float sumOfWeights;
    unsigned int NEvents;
};

#endif
