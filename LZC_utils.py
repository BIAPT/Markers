######## #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
"""
Code used for calculation of Lempel-Ziv complexity. 
This code is using the lempel-ziv implementation from the antropy package [https://raphaelvallat.com/antropy/build/html/index.html]
It only uses the counting part of this package (implementation of Lempel, A., & Ziv, J. (1976).)
The same algorithm can be also found in the Neurokit 2 pakage. [https://neuropsychology.github.io/NeuroKit]

What this code is adding is the use of the LZC with normalization using 
    1) ramdomly shuffled binary timeseries
    2) Phase-shuffled signal (keeping the same spectral properties but shuffeling the phase, used for example by Schartner et al. (2017) )

By questions please ask Charlotte.Maschke@mail.mcgill.ca
"""
######## #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

#!/usr/bin/env python
import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft, ifft
import antropy as ant

def binarize(Ts):
    '''
    Input: continuous unidimensional time series
    Output: binrized time series using absolute value of signal's instantaneous amplitude 
            = absolute value of analtic signal
         array of strings '0' and '1'
    '''
    Thr=np.mean(abs(hilbert(Ts)))
    binary = np.zeros(len(Ts))
    binary[Ts>Thr] = 1

    binary = binary.astype(int)
    # convert to string
    # string = "".join(list(symbolic.astype(int).astype(str)))
    
    return binary

def calculate_LZC(Ts, norm, nsurr = 100):
    '''
    Compute LZc of window:
        1) Binarization using instantaneous amplitude
        2) Calculate raw LZC
        3) normalizes with given method (see below)

    INPUT
        Ts:   Time series data of one channel and epoch as a 1D np array
        norm: Type of normalization can be:
                'shuffle' = for shuffling of the binarized time series
                'phaserand'= for normalization with phase randomized signal
        nsurr: number of surrogates to compute

    OUTPUT
        LZC_norm = normalized Complexity value
        LZC_raw = UNnormalized Complexity value
        C_surr = complexity surrogates shape (1, s_surr)    
    '''
    # binarize signal
    Ts_bin=binarize(Ts)
    # convert to string
    Ts_string = "".join(list(Ts_bin.astype(int).astype(str)))
    LZC_raw = ant.lziv_complexity(Ts_string, normalize=False)
    
    # normalize with surrogate data
    if norm == 'shuffle':
        C_surr = []
        for surr in range(nsurr):
            # shuffle binary time series (applied direclty to Ts_bin)
            np.random.shuffle(Ts_bin)
            Ts_string = "".join(list(Ts_bin.astype(int).astype(str)))
            C_surr.append(ant.lziv_complexity(Ts_string, normalize=False))

    elif norm == 'phaserand':
        C_surr = []
        for surr in range(nsurr):
            # re-create original Ts with random phases:
            ts_phr = phaseRand(Ts)
            # binarize signal
            Ts_bin=binarize(ts_phr)
            # convert to string
            Ts_string = "".join(list(Ts_bin.astype(int).astype(str)))
            C_surr.append(ant.lziv_complexity(Ts_string, normalize=False))
    
    else:
        raise NotImplementedError("not yet implemented, please select norm 'shuffle' or 'phaserand' ")

    LZC_norm = LZC_raw/np.mean(C_surr)
    return LZC_norm, LZC_raw, C_surr

def phaseRand(ts):
    """
    Function to get a phase-randomized version of the time series with maintained spectral properties
    Inspired by matlab version code by Dmytro Iatsenko & Gemma Lancaster
    %  "Surrogate data for hypothesis testing in physical systems", G. Lancaster, et al. Physics Reports, 2018.

    INPUT:
    ts: 1D array of time series signal of one channel
    
    Returns:
    surr_ts: Real valued phase shuffled time series
    
    """
    L=len(ts)
    L2=int(np.ceil(L/2))
    a=0
    b=2*np.pi
    eta=(b-a)*np.random.rand(1,L2-1)+a; # get randomized phases for half of the phases
    # Fourier transform the signal
    ftsig = fft(ts)
    F=ftsig[1:L2] # get first half of signal
    ftsig[1:L2]=F*(np.exp(1j*eta));
    ftsig[1+L-L2:L]=np.conj(np.flip(ftsig[1:L2]));
    #surr_ts=irfft(ftsig.real)
    surr_ts=ifft(ftsig)
    surr_ts = surr_ts.real # double check this
    return surr_ts