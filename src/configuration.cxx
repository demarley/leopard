/*
Created:        14 August 2018
Last Updated:   14 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Configuration class
  -- Read config file and use functions
     to return configurations later
*/
#include "Analysis/leopard/interface/configuration.h"


configuration::configuration(const std::string &configFile) : 
  m_configFile(configFile),
  m_isMC(false),
  m_isTtbar(false),
  m_useTruth(false),
  m_fileInspected(false),
  m_selection("SetMe"),
  m_cutsfile("SetMe"),
  m_treename("SetMe"),
  m_filename("SetMe"),
  m_primaryDataset("SetMe"),
  m_NTotalEvents(0),
  m_verboseLevel("SetMe"),
  m_nEventsToProcess(0),
  m_firstEvent(0),
  m_outputFilePath("SetMe"),
  m_customFileEnding("SetMe"),
  m_makeNewFile(false),
  m_makeHistograms(false),
  m_cma_absPath("SetMe"),
  m_metadataFile("SetMe"),
  m_DNNtraining(false),
  m_DNNinference(false),
  m_dnnFile("SetMe"),
  m_dnnKey("SetMe"){
    m_map_config.clear();
  }

configuration::~configuration() {}

configuration &configuration::operator=(const configuration &rhs) { return *this; }

void configuration::initialize() {
    /* Initialize the configurations */
    std::vector<std::string> configurations; 
    cma::read_file( m_configFile, configurations ); // read config file into vector

    // fill map with values from configuration file
    for (const auto& config : configurations){
        // split config items by space
        std::istringstream cfg(config);
        std::istream_iterator<std::string> start(cfg), stop;
        std::vector<std::string> tokens(start, stop);

        m_map_config.insert( std::pair<std::string,std::string>(tokens.at(0),tokens.at(1)) );
    }

    // Protection against default settings missing in custom configuration
    // -- map of defaultConfigs defined in header (can't use 'verbose' tools, not defined yet!)
    for (const auto& defaultConfig : m_defaultConfigs){
        if ( m_map_config.find(defaultConfig.first) == m_map_config.end() ){ // item isn't in config file
            std::cout << " WARNING :: CONFIG : Configuration " << defaultConfig.first << " not defined" << std::endl;
            std::cout << " WARNING :: CONFIG : Setting value to default " << defaultConfig.second << std::endl;
            m_map_config[defaultConfig.first] = defaultConfig.second;
        }
    }


    // Set the verbosity level (the amount of output to the console)
    std::map<std::string,unsigned int> verboseMap = cma::verboseMap(); // load mapping of string to integer
    m_verboseLevel = getConfigOption("verboseLevel");
    if (verboseMap.find(m_verboseLevel)==verboseMap.end()){
        m_verboseLevel = "INFO";
        cma::setVerboseLevel(m_verboseLevel);

        cma::WARNING( "CONFIG : Verbose level selected, "+m_verboseLevel+", is not supported " );
        cma::WARNING( "CONFIG : Please select one of the following: " );
        for (const auto& dm : verboseMap)
            cma::WARNING( "CONFIG :          "+dm.first);
        cma::WARNING( "CONFIG : Continuing; setting verbose level to "+m_verboseLevel);
    }
    else{
        cma::setVerboseLevel(m_verboseLevel);
    }


    // Get the absolute path to leopard for loading
    char* cma_path = getenv("LEOPARDDIR");
    if (cma_path==NULL){
        cma::WARNING("CONFIG : environment variable " );
        cma::WARNING("CONFIG :    'LEOPARDDIR' " );
        cma::WARNING("CONFIG : is not set.  Using PWD to set path." );
        cma::WARNING("CONFIG : This may cause problems submitting batch jobs." );
        cma_path = getenv("PWD");
    }
    m_cma_absPath = cma_path;
    cma::DEBUG("CONFIG : path set to: "+m_cma_absPath );

    // Assign values
    m_nEventsToProcess = std::stoi(getConfigOption("NEvents"));
    m_firstEvent       = std::stoi(getConfigOption("firstEvent"));
    m_selection        = getConfigOption("selection");
    m_outputFilePath   = getConfigOption("output_path");
    m_customFileEnding = getConfigOption("customFileEnding");
    m_cutsfile         = getConfigOption("cutsfile");
    m_useTruth         = cma::str2bool( getConfigOption("useTruth") );
    m_makeNewFile      = cma::str2bool( getConfigOption("makeNewFile") );
    m_makeHistograms   = cma::str2bool( getConfigOption("makeHistograms") );
    m_dnnFile          = getConfigOption("dnnFile");
    m_dnnKey           = getConfigOption("dnnKey");
    m_DNNtraining      = cma::str2bool( getConfigOption("DNNtraining") );
    m_DNNinference      = cma::str2bool( getConfigOption("DNNinference") );
    m_metadataFile     = getConfigOption("metadataFile");

    cma::read_file( getConfigOption("inputfile"), m_filesToProcess );
    //cma::read_file( getConfigOption("treenames"), m_treeNames );
    m_treename = getConfigOption("treename");

    m_mapOfSamples.clear();
    cma::getSampleWeights( m_metadataFile,m_mapOfSamples );

    return;
}


void configuration::print(){
    // -- Print the configuration
    std::cout << " ** Leopard ** " << std::endl;
    std::cout << " --------------- " << std::endl;
    std::cout << " CONFIGURATION :: Printing configuration " << std::endl;
    std::cout << " " << std::endl;
    for (const auto& config : m_map_config){
        std::cout << " " << config.first << "\t\t\t" << config.second << std::endl;
    }
    std::cout << " --------------- " << std::endl;

    return;
}


std::string configuration::getConfigOption( std::string item ){
    /* Check that the item exists in the map & return it; otherwise throw exception  */
    std::string value("");

    try{
        value = m_map_config.at(item);
    }
    catch(const std::exception&){
        cma::ERROR("CONFIG : Option "+item+" does not exist in configuration.");
        cma::ERROR("CONFIG : This does not exist in the default configuration either.");
        cma::ERROR("CONFIG : Returing an empty string.");
    }

    return value;
}


void configuration::inspectFile( TFile& file ){
    /* Compare filenames to determine file type */
    m_fileInspected = true;
    m_isTtbar = false;
    m_primaryDataset = "";
    m_NTotalEvents = 0;

    // check if file is ttbar
    for (const auto& ttbar : m_ttbarFiles){
        m_isTtbar = (m_filename.find(ttbar)!=std::string::npos);
        m_primaryDataset = ttbar;
        if (m_isTtbar) break;
    }

    m_isMC = (m_isTtbar);

    // get the metadata
    if (m_primaryDataset.size()>0) m_NTotalEvents = m_mapOfSamples.at(m_primaryDataset).NEvents;
    else{
        cma::WARNING("CONFIGURATION : Primary dataset name not found, checking the map");
        for (const auto& s : m_mapOfSamples){
            Sample samp = s.second;
            std::size_t found = m_filename.find(samp.primaryDataset);
            if (found!=std::string::npos){ 
                m_primaryDataset = samp.primaryDataset;
                m_NTotalEvents   = samp.NEvents;
                break;
            } // end if the filename contains this primary dataset
        } // end loop over map of samples (to access metadata info)
    } // end else


    // Protection against accessing truth information that may not exist
    if (!m_isMC && m_useTruth){
        cma::WARNING("CONFIGURATION : 'useTruth=true' but 'isMC=false'");
        cma::WARNING("CONFIGURATION : Setting 'useTruth' to false");
        m_useTruth = false;
    }

    return;
}


void configuration::setTreename(std::string treeName){
    m_treename = treeName;
    return;
}

void configuration::setFilename(std::string fileName){
    m_filename = fileName;
    return;
}

bool configuration::isMC(){
    if (!m_fileInspected){
        cma::WARNING("CONFIGURATION : File not inspected to determine if is MC or data - inspecting now.");
        cma::WARNING("CONFIGURATION : Please call 'isMC( file )' with the ROOT file passed as the argument");
        return false;
    }

    return m_isMC;
}

bool configuration::isMC( TFile& file ){
    /* Check the sum of weights tree DSIDs (to determine Data || MC) */
    inspectFile( file );
    return m_isMC;
}

// THE END
