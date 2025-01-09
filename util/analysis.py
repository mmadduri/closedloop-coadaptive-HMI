import numpy as np
import matplotlib.pyplot as plt
import warnings
import unittest
from scipy import signal, fft
from scipy.stats import wilcoxon
import sklearn
from sklearn import linear_model, metrics


'''
METRIC CALCULATIONS
'''
def calc_rms(signal, remove_offset=True):
    '''
    Root mean square of a signal
    
    Args:
        signal (nt, ...): voltage along time, other dimensions will be preserved
        remove_offset (bool): if true, subtract the mean before calculating RMS
    Returns:
        float array: rms of the signal along the first axis. output dimensions will be the same non-time dimensions as the input signal
    '''
    if remove_offset:
        m = np.mean(signal, axis=0)
    else:
        m = 0
    
    return np.sqrt(np.mean(np.square(signal - m), axis=0))

def calc_time_domain_error(X, Y, axis = 1):
    """calc_time_domain_error

    Args:
        X (n_time x n_dim): time-series data of position, e.g. reference position (time x dimensions)
        Y (n_time x n_dim): time-series data of another position, e.g. cursor position (time x dimensions)
        axis (int): axis to calculate the Euclidean distance along

    Returns:
        td_error (n_time x 1): time-series data of the Euclidean distance between X position and Y position
    """
    # make sure that the shapes are the same
    assert(X.shape == Y.shape)
    td_error = np.linalg.norm(X - Y, axis=axis)
    return td_error


def calc_rel_error(last, first):
    assert(last.shape == first.shape)
    rel_error = (last - first)/first * 100
    return rel_error


'''
TRIAL VELOCITY AND POSITION 
'''

def calculate_intended_vels(ref, pos, fs):
    '''
    ref = 1 x 2
    pos = 1 x 2
    fs = number
    '''
    
    # numbers from Github code
    gain = 120
    ALMOST_ZERO_TOL = 0.01

    intended_vector = (ref - pos)/fs
    
    # in case this is close to zero
    if np.linalg.norm(intended_vector) <= ALMOST_ZERO_TOL:
        intended_norm = np.zeros((2,))
    else:
        intended_norm = intended_vector * gain 
    
    return intended_norm
    

def reconstruct_trial(ref_tr, emg_tr, Ds_tr, time, n_dim = 2, n_ch = 64, fs = 60):
    '''
    reconstruct the cursor position and velocity of a single trial in the continuous-tracking task

    inputs:
    ref_tr = time x 2, reference position
    emg_tr = time x 64, emg during trial
    Ds_tr = time x (2 x 64), decoders during trial
    time = time, length of time to reconstruct. Can use whole trial or portion

    outputs:
    vel_est = time x 2, reconstructed cursor velocity
    pos_est = time x 2; reconstructed cursor position
    int_vel_est = time x 2; reconstrcuted intended velocity

    '''

    assert(ref_tr.shape == (time, n_dim))
    assert(emg_tr.shape == (time, n_ch))
    assert(Ds_tr.shape == (time, n_dim, n_ch))


    time_x = time
    vel_est = np.zeros_like((ref_tr))
    pos_est = np.zeros_like((ref_tr))
    int_vel_est = np.zeros_like((ref_tr))


    hit_bound = 0
    vel_est[0] = Ds_tr[0]@emg_tr[0] 
    pos_est[0] = [0, 0]
    for tt in range(1, time_x):
        vel_plus = Ds_tr[tt-1]@emg_tr[tt] # at time tt
        p_plus = pos_est[tt-1, :] + (vel_est[tt-1, :]/fs)

        # x-coordinate
        if abs(p_plus[0]) > 36:
            p_plus[0] = pos_est[tt-1, 0]
            vel_plus[0] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter

        if abs(p_plus[1]) > 24:
            p_plus[1] = pos_est[tt-1, 1]
            vel_plus[1] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter

        if hit_bound > 200:
            p_plus[0] = 0
            vel_plus[0] = 0
            p_plus[1] = 0
            vel_plus[1] = 0
            hit_bound = 0


        # now update velocity and position
        vel_est[tt] = vel_plus
        pos_est[tt] = p_plus

        # calculate intended velocity
        int_vel_est[tt] = calculate_intended_vels(ref_tr[tt], p_plus, 60)

    return vel_est, pos_est, int_vel_est

def mean_and_interquartile(data, axis = 0):
    """
    calculate mean and interquartile range of data.
    how to use: 
    mean, quantile25, quantile50, quantile75 = mean_and_interquartile(data, axis)
    """
    mean = np.nanmean(data, axis = axis)
    q25, q50, q75 = np.nanpercentile(data, [25, 50, 75], axis=axis)
    return mean, q25, q50, q75


def WilcoxonTest(all_data):
    """Wilcoxon signed-rank test:  
    tests the null hypothesis that two related paired samples come from the same distribution
    """

    n = len(all_data)
    w = np.zeros(n**2)
    p = np.zeros(n**2)
    sig = np.zeros(n**2)
    flag = []
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                w[k], p[k] = wilcoxon(all_data[i],all_data[j])
                # determine significant (p<=0.05)
                if p[k] <= 0.05:
                    sig[k] = 1
            else: 
                w[k] = np.nan
                p[k] = np.nan
                sig[k] = np.nan #i=j
            
            if sig[k] == 1 and j > i:
                flag.append('there is significant difference between condition '+str(i)+' and condition '+str(j)+', w = '+str(w[k])+ ', pvalue = '+str(p[k]))
            
            k+=1    
    return w,p,sig,flag


'''
Time-domain analysis for encoders
'''
    
# EMG = W*v --> encoder 
# trying to estimate W
# xdata = intended velocity (20 seconds)
# ydata = EMG (20 seconds)
def estimate_encoder_linear(x_data, y_data, n_ch = 64, verbose = False): 
    '''
    x_data = (n_samples, n_features) - t x 2 = (n_time, n_dim)
    y_data = (n_samples, n_targets) - t x 64 = (n_time, n_ch)
    weights = (n_targets, n_features) - 64 x 2 = (n_ch, n_dim)
    '''
    regr = linear_model.LinearRegression()
    
    n_time, n_dim = x_data.shape
    # n_ch = y_data.shape[1]
    
    if verbose:
        print("n_time, n_dim, n_ch: ", n_time, n_dim, n_ch)
    # fit y = Wx + b
    regr.fit(x_data, y_data)
    
    # find the W here
    weights = regr.coef_ # 64 x 2
    assert(weights.shape == (n_ch, n_dim))

    if verbose:
        print("weights shape = ", weights.shape)

    # find the b here 
    intercept = regr.intercept_

    if verbose:
        print("intercept shape = ", intercept.shape)

    
    # confirm that the weights match the prediction 
    y_est = regr.predict(x_data)
    y_est_ = (weights@x_data.T + intercept[:, np.newaxis]).T
    
    assert(np.allclose(y_est, y_est_))
    assert(np.allclose(y_est_, y_est))

        
    # make sure that y_est is time x 64
    assert(y_est.shape == (n_time, n_ch))
    
    # what's the r^2 of W?
    # EMG = W*u --> W is encoder, u is the user input
    # v = D*EMG --> decoder
    
    # found by:
    # (1) y_est = regr.predict(x_data) 
    # (2) sklearn.metrics.r2_score(y_data, y_est)
    r2_avg = regr.score(x_data, y_data) # r^2 of the 64 x 2
     
    return (weights, intercept, r2_avg)

def subtract_angles(angle1, angle2):
    """
    Subtract angle2 from angle1, accounting for wraparound.

    Parameters:
        angle1, angle2: NumPy arrays of angles in degrees

    Returns:
        Resulting angles in degrees (NumPy array) within the range [0, 360).
    """
    result = angle1 - angle2
    result %= 360  # Use modulo to ensure the result is within [0, 360)
    result[result > 180] -= 360  # Adjust if result is greater than 180 degrees
    result[result < -180] += 360  # Adjust if result is less than -180 degrees
    return np.abs(result)



'''
Frequency-domain analysis
'''
def FFT(data,N):
  return fft.fft(data)/N

def IFFT(data,N):
  return (fft.ifft(data)*N).real

def frequency_domain(t,fs):
    """ 
    t = time array
    fs = sampling rate (Hz) 
    return:
      xf = positve freq array, length = half of the time array
      xf_all = full freq array, length = same as time array

    """
    N = len(t)                          #data length (time (s) * sampling rate)
    xf_all = fft.fftfreq(N, 1./ fs)     #freq (x-axis) both + and - terms
    xf = fft.fftfreq(N, 1./ fs)[:N//2]  #freq (x-axis) positive-frequency terms
    return xf, xf_all

