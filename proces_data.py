#%%
# Load modules

import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt
import pywt

# Functions
def rotation(df):
    '''
    Rotates acceleration  to 0, 0, 1
    '''
    meanAcceleration = np.mean(df[['AccV', 'AccML', 'AccAP']],axis = 0)
    x_tmp=np.array([0, 0, 1])
    y=np.cross(meanAcceleration,x_tmp)
    x=np.cross(y,meanAcceleration)
    normx=x/np.linalg.norm(x)
    normy=y/np.linalg.norm(y)
    normz= meanAcceleration / np.linalg.norm(meanAcceleration)
    R=np.array((normx.transpose(), normy.transpose(), normz.transpose()))
    accRotated=(R.transpose().dot(df[['AccV', 'AccML', 'AccAP']].transpose())).transpose()
    tmp = pd.DataFrame(accRotated, columns = ['AccAP_rot', 'AccML_rot', 'AccV_rot'])
    return df.join(tmp)

def filt_high(input_signal,  fs, cutoff_h , order =1):
    '''
    Butterworth highpass filter
    '''
    b, a = signal.butter(order, (cutoff_h / (fs / 2)), 'high')
    return signal.filtfilt(b,a, input_signal)

def filt_low(input_signal,  fs, cutoff_l , order =1):
    '''
    Butterworth lowpass filter
    '''
    b, a = signal.butter(order, (cutoff_l / (fs / 2)), 'low')
    return signal.filtfilt(b,a, input_signal)

def normalisation(input_signal):
    # min score: -4G
    # Max score: 4G
    return (input_signal -- 4) / (4 -- 4) * 2 - 1


#%%

for file in os.listdir('Raw_data/train/defog'):
    # Load defog
    acceleration = pd.read_csv(f'Raw_data/train/defog/{file}')

    # Drop unambiguous data
    acceleration = acceleration.loc[(acceleration['Valid'] == True) & (acceleration['Task'] == True)]
    acceleration.reset_index(inplace=True, drop=True)

    # Rotate data
    acceleration = rotation(acceleration)

    # Filter data in all directions
    acceleration['AccV_filt'] = filt_high(acceleration['AccV_rot'], fs = 100 , cutoff_h = 0.01)
    acceleration['AccV_filt'] = filt_low(acceleration['AccV_filt'], fs = 100 , cutoff_l = 10)
    acceleration['AccML_filt'] = filt_high(acceleration['AccML_rot'], fs = 100 , cutoff_h = 0.01)
    acceleration['AccML_filt'] = filt_low(acceleration['AccML_filt'], fs = 100 , cutoff_l = 10)
    acceleration['AccAP_filt'] = filt_high(acceleration['AccAP_rot'], fs = 100 , cutoff_h = 0.01)
    acceleration['AccAP_filt'] = filt_low(acceleration['AccAP_filt'], fs = 100 , cutoff_l = 10)

    # Normalise signal to be between -1, 1
    acceleration['AccV_norm'] = normalisation(acceleration['AccV_filt'])
    acceleration['AccML_norm'] = normalisation(acceleration['AccML_filt'])
    acceleration['AccAP_norm'] = normalisation(acceleration['AccAP_filt'])

    # plot remaining data
    # fig, ax = plt.subplots()
    # ax = plt.plot(acceleration['AccV_norm'])

    #  Wavelet transform
    wavelet = "morl" # mother wavelet
    scales = np.arange(1, 32)

    coeffs_v, freqs = pywt.cwt(acceleration['AccV_norm'], scales, wavelet = wavelet)
    coeffs_v = pd.DataFrame(coeffs_v.T)
    coeffs_ML, freqs = pywt.cwt(acceleration['AccML_norm'], scales, wavelet = wavelet)
    coeffs_ML = pd.DataFrame(coeffs_ML.T)
    coeffs_AP, freqs = pywt.cwt(acceleration['AccAP_norm'], scales, wavelet = wavelet)
    coeffs_AP = pd.DataFrame(coeffs_AP.T)

    for i in coeffs_AP.columns:
        acceleration[f'coeff_V_{i}'] = coeffs_v[i]
        acceleration[f'coeff_ML_{i}'] = coeffs_ML[i]
        acceleration[f'coeff_AP_{i}'] = coeffs_AP[i]

    acceleration.to_csv(f'Processed_data/Train/Defog/{file}')
# %%
