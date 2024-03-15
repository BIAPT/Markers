######## #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
"""
Code used for calculation of spectral power and posterior-anterior ratio.

The posterior-anterior ratio was inspored by a paper by Colombo et al. 2023 [https://academic.oup.com/cercor/article/33/11/7193/7091601]

By questions please ask Charlotte.Maschke@mail.mcgill.ca
"""
######## #### #### #### #### #### #### #### #### #### #### #### #### #### ####  


from scipy.stats import gmean
import numpy as np


def p_a_ratio(epochs, values, anterior_el, posterior_el):
    """
    Calculate the posterior-anterior ratio

    INPUT:
    epochs -- mne epochs, needed to extract channel names of the given data
    values -- values to calculate the p_a_aratio on, np.array with length = n_channels where value[i] corresponds to channel[i]
                ! values must be positive! 
    anterior_el -- list of names of all anterior electrodes (depends on system used), should follow the exact naming of ch_names in eeg.epochs
    posterior_el -- list of names of all anterior electrodes (depends on system used), should follow the exact naming of ch_names in eeg.epochs

    OUTPUT: 
    p_a_ratio -- posterior anterior ratio with values closer to 0 indicating more anterior values dominating posterior values.
                 values of 1 indicaing equal distribution of values.   
    """

    # find channels in data which match posterior and anterior channels
    channels = epochs.ch_names
    idx_anterior = [x in anterior_el for x in channels]
    idx_posterior = [x in posterior_el for x in channels]
    
    # exclude nans from anterior and posterior values
    if np.isnan(values).any():
        anterior = values[idx_anterior][~np.isnan(values[idx_anterior])]
        posterior = values[idx_posterior][~np.isnan(values[idx_posterior])]

    else:
        anterior = values[idx_anterior]
        posterior = values[idx_posterior]

    # calculate ratio of geometric mean
    p_a_ratio = gmean(posterior) / gmean(anterior)

    return p_a_ratio
