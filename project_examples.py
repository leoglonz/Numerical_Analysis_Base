# For efficiency calculations.
import numpy as np
import time
import project_lonzarich as num

# For example using LIGO Event data.
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import readligo as rl # For reading LIGO files.
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


efficiency = 0
example = 1



''' ****** Calculating Efficiency of FFT Versus Naive DFT ****** '''

if (efficiency == 1):
    #Creating random vector x.
    n = 2**15
    k = 1
    x = np.random.rand(n,1)

    t1_total = 0
    t2_total = 0


    # Calculating to remove anomaly values. See below.
    num.dft(x)
    np.fft.fft(x)

    # Taking average of time needed to compute each transform.
    for i in range(k):
        t = time.process_time()
        num.dft(x)
        t1_total += time.process_time() - t

        t = time.process_time()
        np.fft.fft(x)
        t2_total += time.process_time() - t

    # For some reason the first computations of the DFT and FFT were always
    # slower/ faster than those proceeding (I attribute this to Python or Numpy
    # storing repeated values, though, testing this, it did not seem to be the
    # case. I have not yet been able to find any documentation on this
    # phenomenon).

    # To account for this, are initially computed so that the average does not
    # pick up the anomaly values.

    t1_avg = t1_total/k

    t2_avg = t2_total/k

    print('Avg. Time for DFT: ', t1_avg)
    print('Avg. Time for FFT: ', t2_avg)




'''__________________________________________________________________________
    ****** Example Problem: Binary Black Hole Signals: (OS) LIGO Data ******
   __________________________________________________________________________'''

# Data and packages Courtesy of Gravitational Wave Open Science Center (GWOSC).

# R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration),
# "Open data from the first and second observing runs of Advanced LIGO and
# Advanced Virgo", arxiv:1912.11716


# Signal processing Example using LIGO Event GW150914 from the LIGO WV Open
# Science Center.

# Importing event data using GWOSC package (This is an observation of
# gravitational waves from a binary black hole merger -- in this case, two
# gravitationally bound black holes collapse on one another. This is the
# largest known source of gravitational waves).

if (example == 1):
    events = json.load(open("BBH_events_v3.json","r"))
    event = events['GW150914']

    # Extract the parameters for the desired event:
    fn_L = event['fn_L1']              # File name for Livingston detector data
    sample_freq = event['fs']           # Data sampling rate
    tevent = event['tevent']            # Set approximate event in GPS time



    # Reading intensity and time data from Livingston using GWOSC package.
    L_mag, time, dictionary = rl.loaddata(fn_L, 'L1')



    # Setting the radius of delta_t on the amount of time we want to select points
    # from around the merger.
    delta_t = 5
    index = (time < tevent + delta_t) & (time >= tevent - delta_t)


    # Plotting signal intensity vs time.
    plt.figure()
    plt.plot(time[index] - tevent, L_mag[index], c = 'b')
    plt.xlabel('Time (s) before and after merger')
    plt.ylabel('Intensity')
    plt.title('Intensity for Black Hole Merger GW150914')
    plt.grid('on')
    plt.savefig('Intensity.')

    plt.show()


    # Number of points being used, and defining frequency min/max (these are
    # selected due to limitations in equipment used to collect the data).
    n = 3 * sample_freq
    freq_min = 20.
    freq_max = 2000.

    # Finding power spectrum density using the FFT and Welch's method.
    psd_L, freqs = mlab.psd(L_mag, Fs = sample_freq, NFFT = n)

    # To keep values manageable without removing frequency information,
    psd_L = np.sqrt(psd_L)

    # Plotting power spectrum densities to determine dominant frequencies.
    plt.figure(figsize=(8,6))
    plt.loglog(freqs, psd_L, c = 'r')
    plt.axis([freq_min, freq_max, 1e-24, 1e-19])

    plt.ylabel('Amplitude Spectral density ($H^{-1/2}$)')
    plt.xlabel('Frequency (Hz)')
    plt.title('PSD for Black Hole Merger GW150914')
    plt.grid('on')
    plt.savefig('PSD')

    plt.show()

'''
    # Use the FFT and PSD to filter out noise.
    n = (131072/4096) * sample_freq
    fft = np.fft.fft(L_mag, n)
    PSD = fft * np.conj(fft)/n

    freq = (1/(dt * n)) * np.arange(n)

    # Finding frequencies with large powers (representations).
    # Lower bound observed from PSD plot:

    indicies = PSD > 10**(-21)

    # Sets all unwanted frequency signals to zero
    clean_signal = PSD * indicies

    fft = indicies * fft

    # Inverse to find clean signal.
    clean_signal = np.fft.ifft(fft)

    # Plotting signal intensity vs time.
    plt.figure()
    plt.plot(time, clean_signal, c = 'b')
    plt.xlabel('Time (s) before and after merger')
    plt.ylabel('Intensity')
    plt.title('Intensity for Black Hole Merger GW150914')
    plt.grid('on')
    plt.savefig('Intensity.')
'''
