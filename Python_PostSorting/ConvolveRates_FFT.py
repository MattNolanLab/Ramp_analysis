import numpy as np



def nextpow2(x):
    """ Return the smallest integral power of 2 that >= x """
    n = 2
    while n < x:
        n = 2 * n
    return n


def fftkernel(x, w):
    """
    y = fftkernel(x,w)
    Function `fftkernel' applies the Gauss kernel smoother to an input
    signal using FFT algorithm.
    Input argument
    x:    Sample signal vector.
    w: 	Kernel bandwidth (the standard deviation) in unit of
    the sampling resolution of x.
    Output argument
    y: 	Smoothed signal.
    MAY 5/23, 2012 Author Hideaki Shimazaki
    RIKEN Brain Science Insitute
    http://2000.jukuin.keio.ac.jp/shimazaki
    Ported to Python: Subhasis Ray, NCBS. Tue Jun 10 10:42:38 IST 2014
    """
    L = len(x)
    Lmax = L + 3 * w
    n = nextpow2(Lmax)
    X = np.fft.fft(x, n)
    f = np.arange(0, n, 1.0) / n
    f = np.concatenate((-f[:int(n / 2)], f[int(n / 2):0:-1]))
    K = np.exp(-0.5 * (w * 2 * np.pi * f)**2)
    y = np.fft.ifft(X * K, n)
    y = y[:L].copy()
    return y



def convolve_binned_spikes(binned_spike_times):
    convolved_spikes=[]
    convolved_spikes = fftkernel(binned_spike_times, 2)
    return convolved_spikes.real

