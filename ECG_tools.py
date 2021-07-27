import sys
import os.path, time 
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

# included with carpentry or openCARP https://opencarp.org/
from carputils.carpio import igb

def get_torso_electrode_phie(phie_data, elec_name):
  ''' Get Phie over time at the torso electrode locations. 
      Only for torso models with electrodes within the mesh.  
    Parameters:
      phie_data (array): torso Phie data loaded from IGB file
      elec_name (string): name of file with electrode numbers (not coordinates)
    Return:
      elec_data (dict): Phie on ECG electrodes
    
    by Sofia Monaci
  '''

  elec_pts = np.loadtxt(elec_name, usecols=(1,), dtype=int)

  elec_data = {'V1': phie_data[elec_pts[0],:],
               'V2': phie_data[elec_pts[1],:],
               'V3': phie_data[elec_pts[2],:],
               'V4': phie_data[elec_pts[3],:],
               'V5': phie_data[elec_pts[4],:],
               'V6': phie_data[elec_pts[5],:],
               'RA': phie_data[elec_pts[6],:],
               'LA': phie_data[elec_pts[7],:],                  
               'RL': phie_data[elec_pts[8],:],
               'LL': phie_data[elec_pts[9],:]}
  
  return elec_data

    
def convert_electrodes_to_ECG(elec_data):
  ''' Compute 12-lead ECG from electrode data
    Parameters:
      elec_data (dict): Phie on ECG electrodes
    Return:
      ecg (dict): 12-lead ECG 

    by Sofia Monaci
  '''

  # Defines the Wilson Central Terminal
  wct = elec_data['LA'] + elec_data['RA'] + elec_data['LL']

  ecg = dict()
  
  # Precordeal leads
  ecg['V1'] = elec_data['V1'] - wct/3.
  ecg['V2'] = elec_data['V2'] - wct/3.
  ecg['V3'] = elec_data['V3'] - wct/3.
  ecg['V4'] = elec_data['V4'] - wct/3.
  ecg['V5'] = elec_data['V5'] - wct/3.
  ecg['V6'] = elec_data['V6'] - wct/3.

  # Limb Leads
  ecg['LI'] = elec_data['LA'] - elec_data['RA']
  ecg['LII'] = elec_data['LL'] - elec_data['RA']
  ecg['LIII'] = elec_data['LL'] - elec_data['LA']

  # Augmented leads
  ecg['aVR'] = elec_data['RA'] - 0.5*(elec_data['LA'] + elec_data['LL'])
  ecg['aVL'] = elec_data['LA'] - 0.5*(elec_data['RA'] + elec_data['LL'])
  ecg['aVF'] = elec_data['LL'] - 0.5*(elec_data['LA'] + elec_data['RA'])

  return ecg


def get_torso_ecg(phie_file, elec_name):
  ''' Create a 12-lead ECG from torso electrode data including time step data
    Parameters:
      elec_data (dict): Phie on torso electrodes
    Return:
      ecg (dict): 12-lead ECG 

    by Sofia Monaci
  '''
      
    # Gets data from phie file
    phie_data, t_steps = cp.igbread(phie_file)
    
    # Extracts electrode potentials from phie file with electrode numbers file
    elec_data = get_torso_electrode_phie(phie_data, elec_name)
    
    # Converts potential electrode data into full 12-lead ECG
    ecg = convert_electrodes_to_ECG(elec_data)
    
    return ecg, t_steps


def write_ecg_to_csv(file_name, ecg_data):
  ''' Write out 12-lead ECG data from ecg dict to CSV file
    Parameters: 
      file_name (string): ECG file name
      ecg_data (dict): 12-lead ECG
      
    by Sofia Monaci
  '''
  
    with open(file_name, mode='w') as csv_file:
        field_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LI', 'LII', 'LIII', 'aVR', 'aVL', 'aVF']
        cw = csv.writer(csv_file)  

        for row in field_names:
            cw.writerow(ecg_data[row])


def read_ecg_from_csv(file_name):
  ''' Read 12-lead ECG data from CSV file
      Saved with write_ecg_to_csv()
    Parameters: 
      file_name (string): ECG file name
    Return:
      ecg_data (dict): 12-lead ECG
      t_steps (array): time step data
      
    by Sofia Monaci
  '''
    
    data = np.genfromtxt(file_name, delimiter = ',')
    ecg = {'V1' : data[0],
           'V2' : data[1],
           'V3' : data[2],
           'V4' : data[3],
           'V5' : data[4],
           'V6' : data[5],
           'LI' : data[6],
           'LII' : data[7],
           'LIII' : data[8],
           'aVR' : data[9],
           'aVL' : data[10],
           'aVF' : data[11]}
    
    t_steps = np.size(ecg['V1'])
    
    return ecg, t_steps

def write_ecg_to_data_file(file_name, ecg, t_steps):
  ''' Write ECG leads as numerical data only using Numpy
      Format: tsteps LI LII LIII aVR aVL aVF V1 V2 V3 V4 V5 V6
    Parameters: 
      file_name (string): name of output file
      ecg (dict): dict with ECG data
      t_steps (array): time steps
      
    by Caroline Mendonca Costa
  '''
  # TODO: CHANGE ORDER. CHECK TRANSPOSE
  
  array = np.array([t_steps, 
                    ecg['LI'],
                    ecg['LII'], 
                    ecg['LIII'], 
                    ecg['aVR'], 
                    ecg['aVL'], 
                    ecg['aVF'], 
                    ecg['V1'], 
                    ecg['V2'], 
                    ecg['V3'], 
                    ecg['V4'], 
                    ecg['V5'], 
                    ecg['V6']]).T # CHECK TRANSPOSE. WHY TWO???
  
  np.savetxt(file_name, array.T, delimiter=' ')


def read_ecg_from_precision(file_name):
  ''' Read 12-lead ECG data from Precision's CSV file and store as ecg dictionary
    Parameters:
      file_name (string): name of input file
    Return:
      ecg (dict): dict with ECG data
      t_steps (array): time steps
     
    by Caroline Mendonca Costa
  '''
  
  # skip the first 86 rows - rubbish
  data = pd.read_csv(file_name, header=0, skiprows=86, skipfooter=1, engine='python', sep=',').values.squeeze()
  
  t_steps = data[:,2]
  
  ecg = {'LI'   : data[:,3],
         'LII'  : data[:,6],
         'LIII' : data[:,9],
         'aVR'  : data[:,12],
         'aVL'  : data[:,15],
         'aVF'  : data[:,18],
         'V1'   : data[:,21],
         'V2'   : data[:,24],
         'V3'   : data[:,27],
         'V4'   : data[:,30],
         'V5'   : data[:,33],
         'V6'   : data[:,36]}    
  
  return ecg, t_steps


def get_ecg_from_file(ecg_file):
  ''' Read ECG leads from data file 
      Assumes format: tsteps LI LII LIII aVR aVL aVF V1 V2 V3 V4 V5 V6 
    Parameters
      ecg_file (string): name of ECG file
    Return:
      ecg (dict): dict with ECG data
      t_steps (array): time steps
      
    by Caroline Mendonca Costa
  '''
  # TODO: CHANGE ECG FORMAT
  
  ecg_data = np.loadtxt(ecg_file, dtype=float)
  
  t_steps = ecg_data[:,0]

  ecg = dict()
  
  ecg['LI'] = ecg_data[:,1]
  ecg['LII'] = ecg_data[:,2]
  ecg['LIII'] = ecg_data[:,3]
  ecg['aVR'] = ecg_data[:,4]
  ecg['aVL'] = ecg_data[:,5]
  ecg['aVF'] = ecg_data[:,6]
  ecg['V1'] = ecg_data[:,7]
  ecg['V2'] = ecg_data[:,8]
  ecg['V3'] = ecg_data[:,9]
  ecg['V4'] = ecg_data[:,10]
  ecg['V5'] = ecg_data[:,11]
  ecg['V6'] = ecg_data[:,12]
  
  return ecg, t_steps


def get_electrodes_from_phierec_file(phie_file):
  ''' Read Phie electrodes from phie recovery file 
      Assumes electrode format: V1 V2 V3 V4 V5 V6 RA LA RL LL
    Parameters:
      phie_file (string): name of phie file
    Return:
      phie_data (dict): dict with phie data
      t_steps (array): time steps
      
    by Caroline Mendonca Costa
  '''
  
  igbobj = igb.IGBFile(phie_file)
  header = igbobj.header()
  data = igbobj.data()
  igbobj.close()
  
  ntraces = header.get('x')
  ntsteps = header.get('t')
  dimt = header.get('dim_t')
  
  # make sure data has correct size - something wrong with phie recovery sim: running more steps than defined (????)
  data_reshaped = np.reshape(data[0:ntsteps*ntraces], (ntsteps,ntraces))

  t_steps = np.linspace(0, dimt, ntsteps)

  phie_data = {'V1': data_reshaped[:,0],
               'V2': data_reshaped[:,1],
               'V3': data_reshaped[:,2],
               'V4': data_reshaped[:,3],
               'V5': data_reshaped[:,4],
               'V6': data_reshaped[:,5],
               'RA': data_reshaped[:,6],
               'LA': data_reshaped[:,7],                  
               'RL': data_reshaped[:,8],
               'LL': data_reshaped[:,9]}
  
  return phie_data, t_steps
  
 
def normalize_ECG(ecg):
  ''' Normalize ECG leads to facilitate comparison between models
    Parameters:
      ecg (dict): 12 lead ECG dictionary
    Return:
      norm_ecg (dict): normalized 12 lead ECG dictionary
  
    by Caroline Mendonca Costa
  '''

    norm_ecg = dict()
    
    # divide by maximum absolute value
    norm_ecg['V1'] = np.divide(ecg['V1'],np.amax(np.absolute(ecg['V1'])))
    norm_ecg['V2'] = np.divide(ecg['V2'],np.amax(np.absolute(ecg['V2'])))
    norm_ecg['V3'] = np.divide(ecg['V3'],np.amax(np.absolute(ecg['V3'])))
    norm_ecg['V4'] = np.divide(ecg['V4'],np.amax(np.absolute(ecg['V4'])))
    norm_ecg['V5'] = np.divide(ecg['V5'],np.amax(np.absolute(ecg['V5'])))
    norm_ecg['V6'] = np.divide(ecg['V6'],np.amax(np.absolute(ecg['V6'])))
    norm_ecg['LI'] = np.divide(ecg['LI'],np.amax(np.absolute(ecg['LI'])))
    norm_ecg['LII'] = np.divide(ecg['LII'],np.amax(np.absolute(ecg['LII'])))
    norm_ecg['LIII'] = np.divide(ecg['LIII'],np.amax(np.absolute(ecg['LIII'])))
    norm_ecg['aVR'] = np.divide(ecg['aVR'],np.amax(np.absolute(ecg['aVR'])))
    norm_ecg['aVL'] = np.divide(ecg['aVL'],np.amax(np.absolute(ecg['aVL'])))
    norm_ecg['aVF'] = np.divide(ecg['aVF'],np.amax(np.absolute(ecg['aVF'])))

    return norm_ecg


def plotECG_QRSd_QT(t_steps, ecg, qrs_start, qrs_end, qt_int, y_lim, show_fig, fig_name):
  ''' Plot all ECG leads with QRSd and QT as horizontal lines
    Parameters:
      t_steps (array): ECG time steps
      ecg (dict): 12 lead ECG 
      qrs_start (float): time QRS complex starts
      qrs_end (float): time QRS complex ends
      qt_int (float): QT interval
      y_lim (array): 2 elements array with y-axis lower and upper limits on plot
      show_fig (bool): show figure on screen before saving
      fig_name (string): name of figure file
      
    by Caroline Mendonca Costa
  '''
  
  # end of QT interval
  qt_end = qrs_start + qt_int

  plt.rcParams['font.size'] = '16'

  fig, ax = plt.subplots() # create figure and axes objects
  lead_names = ecg.keys()
  for lead_name in lead_names:
    ax.plot(t_steps, ecg[lead_name], color='k', linewidth=2)
    
  # adjust axes
  plt.xlim(0, t_steps[len(t_steps)-1])
  plt.ylim(y_lim[0], y_lim[1])
  
  x_int = 100
  y_int = abs(y_lim[1]-y_lim[0])/5.
  ax.set_xticks(np.arange(0, t_steps[len(t_steps)-1]+1, x_int))
  ax.set_yticks(np.arange(y_lim[0], y_lim[1]+y_int, y_int))  
  
  # add vertical lines with QRS start and end, and QT end
  ax.vlines(x=qrs_start, ymin=y_lim[0], ymax=y_lim[1], colors='tab:grey', linestyle='dashed', linewidth=2)
  ax.vlines(x=qrs_end, ymin=y_lim[0], ymax=y_lim[1], colors='tab:grey', linestyle='dashed', linewidth=2)
  ax.vlines(x=qt_end, ymin=y_lim[0], ymax=y_lim[1], colors='tab:grey', linestyle='dashed', linewidth=2)

  plt.xlabel('Time (ms)')
  plt.ylabel('Normalized Voltage')

  ax.grid(alpha=0.5)
  plt.tight_layout()
 
  if show_fig:
    plt.savefig(fig_name)
    plt.show()  
  else:
    plt.savefig(fig_name)
    plt.close() 
