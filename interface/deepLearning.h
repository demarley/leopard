#ifndef DEEPLEARNING_H
#define DEEPLEARNING_H

#include <string>
#include <map>
#include <vector>

#include "lwtnn/lwtnn/interface/LightweightNeuralNetwork.hh"
#include "lwtnn/lwtnn/interface/parse_json.hh"

#include "Analysis/leopard/interface/tools.h"
#include "Analysis/leopard/interface/configuration.h"
#include "Analysis/leopard/interface/physicsObjects.h"


class deepLearning {
  public:
    deepLearning( configuration& cmaConfig );

    ~deepLearning();

    void training(Top& top);
    void inference(Top& top);
    void loadFeatures(const Top& top);

    std::map<std::string,double> predictions();
    double prediction();
    double prediction(const std::string& key);
    std::map<std::string,double> features();

  protected:

    configuration *m_config;

    lwt::LightweightNeuralNetwork* m_lwnn;       // LWTNN tool
    std::map<std::string, double> m_dnnInputs;   // values for inputs to the DNN
    std::string m_dnnKey;                        // default key for accessing map of values
    float m_DNN;                                 // DNN prediction for one key

    std::map<std::string,double> m_discriminant; // map of DNN predictions
};

#endif
