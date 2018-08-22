"""
Extra labels for plots
"""
import os
import sys
from array import array

try:
    CMSSW_BASE = os.environ['CMSSW_BASE']
    from Analysis.hepPlotter.histogram1D import Histogram1D
    from Analysis.hepPlotter.histogram1D import Histogram2D
    import Analysis.hepPlotter.labels as hpl
    import Analysis.hepPlotter.tools as hpt
except KeyError:
    cwd = os.getcwd()
    hpd = cwd.rstrip("leopard/python")+"/hepPlotter/python/"
    if hpd not in sys.path:
        sys.path.insert(0,hpd)
    import tools as hpt


class Sample(object):
    """Class for organizing plotting information about physics samples"""
    def __init__(self,label='',color=''):
        self.label = label
        self.color = color

class Variable(object):
    """Class for organizing plotting information about variables"""
    def __init__(self,binning=[],label=''):
        self.binning = binning
        self.label   = label


def variable_labels():
    """Dictionaries that contain Variables objects."""
    _phi  = r'$\phi$'
    _eta  = r'$\eta$'
    _T    = r'$_{\text{T}}$ [GeV]'
    _mass = 'Mass [GeV]'
    bdisc_bins = array('d',[i*0.1 for i in range(11)])  # default value = -1

    variables = {}

    variables['mass_lep_AK4']   = Variable(binning=hpt.hist1d(32,  0.,  800.),label=r'm$_{\ell+\text{AK4}}$')
    variables['AK4_CSVv2']      = Variable(binning=bdisc_bins, label=r'AK4 CSVv2')
    variables['deltaR_lep_AK4'] = Variable(binning=hpt.hist1d(50,0,5),   label=r'$\Delta$R(Lepton,AK4)')
    variables['ptrel_lep_AK4']  = Variable(binning=hpt.hist1d(20,0,200), label=r'p$_\text{T}^\text{rel}$(Lepton,AK4)')

    variables['jet_pt']  =   Variable(binning=hpt.hist1d(40,  0.,2000.), label=r'Small-R Jet p'+_T)
    variables['jet_eta'] =   Variable(binning=hpt.hist1d(10,-2.5,  2.5), label=r'Small-R Jet '+_eta)
    variables['lep_eta'] = Variable(binning=hpt.hist1d(10,-2.5,   2.5),label=r'Lepton '+_eta)
    variables['lep_pt']  = Variable(binning=hpt.hist1d(10, 25.,  300.),label=r'Lepton p'+_T)
    variables['el_eta']  = Variable(binning=hpt.hist1d(10,-2.5,   2.5),label=r'Electron '+_eta)
    variables['el_pt']   = Variable(binning=hpt.hist1d(10,  0.,  500.),label=r'Electron p'+_T)
    variables['mu_eta']  = Variable(binning=hpt.hist1d(10,-2.5,   2.5),label=r'Muon '+_eta)
    variables['mu_pt']   = Variable(binning=hpt.hist1d(10,  0.,  500.),label=r'Muon p'+_T)
    variables['lepton_eta'] = variables['lep_eta']
    variables['lepton_pt']  = variables['lep_pt']

    variables['mtw']     = Variable(binning=hpt.hist1d(12,  0.,  120.),label=r'$\mathsf{m_T^W}$ [GeV]')
    variables['met_met'] = Variable(binning=hpt.hist1d(50,    0,1000), label=r'E$_{\text{T}}^{\text{miss}}$ [GeV]')
    variables['met_phi'] = Variable(binning=hpt.hist1d(16, -3.2, 3.2), label=r'$\phi^{\text{miss}}$ [GeV]')

    return variables



def sample_labels():
    """Dictionaries that contain Samples objects.
       > The key values match those in config/sampleMetadata.txt.
         (The functions in util.py are setup to read the information this way.)
         If you want something unique, then you may need to specify 
         it manually in your plotting script
    """
    ## Sample information
    samples = {}

    samples['signal'] = Sample(label='Signal',color='b')
    samples['bckg']   = Sample(label='Bckg',color='r')

    # Standard Model
    ttbar = r't$\bar{\text{t}}$'
    samples['ttbar']     = Sample(label=ttbar,color='white')
    samples['dijet']     = Sample(label=r'Dijets', color='purple')
    samples['multijet']  = Sample(label=r'Multi-jet', color='purple')
    samples['diboson']   = Sample(label=r'Diboson',color='green')
    samples['singletop'] = Sample(label=r'Single Top',color='blue')
    samples['ttbarW']    = Sample(label=ttbar+'W',color='#C9FFE5')
    samples['ttbarZ']    = Sample(label=ttbar+'Z',color='#7FFFD4')
    samples['ttbarV']    = Sample(label=ttbar+'V',color='cyan')
    samples['ttbarH']    = Sample(label=ttbar+'H',color='#3AB09E')
    samples['ttbarX']    = Sample(label=ttbar+'V',color='#008B8B')
    samples['wjets']     = Sample(label=r'W+jets',color='yellow')
    samples['zjets']     = Sample(label=r'Z+jets',color='darkorange')

    # Data
    samples['data']      = Sample(label=r'Data',color='black')

    return samples
