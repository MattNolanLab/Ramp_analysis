import numpy as np
from scipy import signal
import matplotlib.pylab as plt
import math
import os
import elephant as elephant

"""
https://elifesciences.org/articles/35949#s4
eLife 2018;7:e35949 DOI: 10.7554/eLife.35949

Kornienko et al., 2018
The theta rhythmicity of neurons was estimated from the instantaneous firing rate of the cell.
The number of spikes observed in 1 ms time window was calculated and convolved with a Gaussian kernel (standard deviation of 5 ms).
The firing probability was integrated over 2 ms windows and transformed into a firing rate.
A power spectrum of the instantaneous firing rate was calculated using the pwelchfunction of the oce R package.
The estimates of the spectral density were scaled by multiplying them by the corresponding frequencies: spec(x)∗freq(x).
A theta rhythmicity index for each neuron was defined as θ−baselineθ+baseline, where θ is the mean power at 6–10 Hz and baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).
The theta rhythmicity indices of HD cells were analyzed with the Mclust function of the R package mclust which uses Gaussian mixture modeling and the EM algorithm to estimate the number of components in the data.



# obtain spike-time autocorrelations
windowSize<-300; binSize<-2
source(paste(indir,"get_stime_autocorrelation.R",sep="/"))
runOnSessionList(ep,sessionList=rss,fnct=get_stime_autocorrelation,
                 save=T,overwrite=T,parallel=T,cluster=cl,windowSize,binSize)
rm(get_stime_autocorrelation)

get_frequency_spectrum<-function(rs){
  print(rs@session)
  myList<-getRecSessionObjects(rs)
  st<-myList$st
  pt<-myList$pt
  cg<-myList$cg

  wf=c();wfid=c()
  m<-getIntervalsAtSpeed(pt,5,100)
  for (cc in 1:length(cg@id)) {
    st<-myList$st
    st<-setCellList(st,cc+1)
  ##########################################################
    st1<-setIntervals(st,s=m)
    st1<-ifr(st1,kernelSdMs = 5,windowSizeMs = 2)
    Fs=1000/2
    x=st1@ifr[1,]
    xts <- ts(x, frequency=Fs)
    w <- oce::pwelch(xts,nfft=512*2, plot=FALSE,log="no")
    wf0=w$spec*w$freq
    wf=rbind(wf,wf0)
    wfid=cbind(wfid,cg@id[cc])
  }
  return(list(spectrum=t(wf),spectrum.id=wfid,spectrum.freq=w$freq))
  }
  
##################################################################################
# calculate theta index from power spectra of instantaneous firing rates
freq=spectrum.freq[1,]
theta.i=c()
for (i in 1:dim(spectrum)[2]){
  wf=spectrum[,i]
  th=mean(wf[freq>6 & freq<10])
  b=mean(c(wf[freq>3 & freq<5],wf[freq>11 & freq<13]))
  ti=(th-b)/(th+b)
  theta.i=c(theta.i,ti)
}

x=theta.i[t$hd==1]
par(mfrow=c(2,3))
hist(x,main="Theta index distribution",ylab = "Number of cells", xlab="Theta index",15,xlim = c(-.05,.4),las=1)

x.gmm = Mclust(x)
x.s=summary(x.gmm)
print("Fit Gaussian finite mixture model")
print(paste("Number of components of best fit: ",x.s$G,sep=""))
print(paste("Log-likelhood: ",round(x.s$loglik,2),sep=""))
print(paste("BIC: ",round(x.s$bic,2),sep=""))
print("Theta index threshold = 0.07")
lines(c(0.07,0.07),c(0,14),col="red",lwd=2)
print(paste("Number of non-rhythmic (NR) HD cells (theta index threshold < 0.07): N = ",sum(x<.07),sep=""))
print(paste("Number of theta-rhythmic (TR) HD cells (theta index threshold > 0.07): N = ",sum(x>.07),sep=""))

##################################################################################
  
"""

def calculate_theta_rythmicity(f, Pxx_den):
    in_theta_power = np.take(Pxx_den, np.where(np.logical_and(f > 8 and f < 13)[1]))
    out_theta_power = np.take(Pxx_den, np.where(np.logical_and(f < 8 and f > 13)[1]))
    return in_theta_power, out_theta_power


def calculate_spectral_density(firing_rate, prm, spike_data, cluster, cluster_index, save_path):
    f, Pxx_den = signal.welch(firing_rate, fs=100, scaling='spectrum')
    print(cluster)
    plt.semilogy(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_power_spectra_ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return f, Pxx_den


def calculate_power_spectrum(firing_rate):
    ps = np.abs(np.fft.fft(firing_rate))**2

    time_step = 1
    freqs = np.fft.fftfreq(firing_rate.size, time_step)
    idx = np.argsort(freqs)

    plt.plot(freqs[idx], ps[idx])
    return freqs[idx], ps[idx]


def calculate_firing_probability(convolved_spikes):
    firing_rate=[]
    firing_rate = get_rolling_sum(convolved_spikes, 2)
    return (firing_rate*1000)/2 # convert to Hz


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:] / window


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def convolve_spikes(spikes):
    window = signal.gaussian(3, std=5)
    convolved_spikes = signal.convolve(spikes, window, mode='full')
    return convolved_spikes


def bin_into_1ms(spike_times, number_of_bins):
    posrange = np.linspace(number_of_bins.min(), number_of_bins.max(),  num=max(number_of_bins)+1)
    values = np.array([[posrange[0], posrange[-1]]])
    H, bins = np.histogram(spike_times, bins=(posrange), range=values)
    return H


def extract_instantaneous_firing_rate(spike_data, cluster):
    firing_times=spike_data.at[cluster, "firing_times"]/30 # convert from samples to ms
    bins = np.arange(0,np.argmax(firing_times), 1)
    instantaneous_firing_rate = bin_into_1ms(firing_times, bins)
    return instantaneous_firing_rate


def calculate_theta_power(Pxx_den,f):
    theta_power = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f > 4, f < 12))))
    baseline = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >  0, f < 50))))
    #adjacent_power2 = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >= 11, f <=13))))
    #baseline = (adjacent_power1 + adjacent_power2)/2
    x = theta_power - baseline
    y = theta_power + baseline
    t_index = x/y
    return t_index


def calculate_theta_index(spike_data,prm):
    print('I am calculating theta index...')
    spike_data["ThetaIndex"] = ""
    save_path = prm.get_output_path() + '/Figures/firing_properties/autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1

        instantaneous_firing_rate = extract_instantaneous_firing_rate(spike_data, cluster)
        #convolved_spikes = convolve_spikes(instantaneous_firing_rate)
        firing_rate = calculate_firing_probability(instantaneous_firing_rate)
        f, Pxx_den = calculate_spectral_density(firing_rate, prm, spike_data, cluster, cluster_index, save_path)

        t_index = calculate_theta_power(Pxx_den, f)

        firing_times_cluster = spike_data.firing_times[cluster]
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 600)

        fig = plt.figure(figsize=(7,6)) # width, height?
        ax = fig.add_subplot(1, 2, 1)  # specify (nrows, ncols, axnum)
        ax.set_xlim(-300, 300)
        ax.plot(time, corr, '-', color='black')
        x=np.max(corr)
        ax.text(-200,x, "theta index = " + str(round(t_index,3)), fontsize =10)

        ax = fig.add_subplot(1, 2, 2)  # specify (nrows, ncols, axnum)
        ax.semilogy(f, Pxx_den)
        plt.ylim(0,20)
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('PSD [V**2/Hz]')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_theta_properties.png', dpi=300)
        plt.close()

    return spike_data




##################################################################################



def calculate_autocorrelogram_hist(spikes, bin_size, window):

    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time



def plot_autocorrelograms(spike_data, prm):
    plt.close()
    print('I will plot autocorrelograms for each cluster.')
    save_path = prm.get_output_path() + '/Figures/firing_properties/autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        firing_times_cluster = spike_data.firing_times[cluster]
        #lags = plt.acorr(firing_times_cluster, maxlags=firing_times_cluster.size-1)
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 20)
        plt.xlim(-10, 10)
        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_autocorrelogram_10ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.figure()
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 500)
        plt.xlim(-250, 250)
        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_autocorrelogram_250ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()




def convolve_array(spikes):
    window = signal.gaussian(2, std=5)
    convolved_spikes = signal.convolve(spikes, window, mode='full')
    return convolved_spikes


#A theta rhythmicity index for each neuron was defined as θ−baselineθ+baseline,
## where θ is the mean power at 6–10 Hz and baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).

"""
The theta index was calculated here as by Yartsev et al., (2011). 
First, we computed the autocorrelation of the spike train binned by 0.01 seconds with lags up to ±0.5 seconds. 
Without normalization, this may be interpreted as the counts of spikes that occurred in each 0.01 second bin 
after a previous spike (Figure 1a). The mean was then subtracted, and the spectrum was calculated as the square 
of the magnitude of the fast-Fourier transform of this signal, zero-padded to 216 samples. 
This spectrum was then smoothed with a 2-Hz rectangular window (Figure 1b), 
and the theta index was calculated as the ratio of the mean of the spectrum within 1-Hz of each side of the 
peak in the 5-11 Hz range to the mean power between 0 and 50 Hz.
"""

##
def calculate_theta_rythmicity(spike_data, prm, save_path):
    print('I am calculating theta index ...')
    spike_data["ThetaIndex"] = ""
    save_path = prm.get_output_path() + '/Figures/firing_properties/power_spectra'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        firing_times_cluster = spike_data.firing_times[cluster]
        #lags = plt.acorr(firing_times_cluster, maxlags=firing_times_cluster.size-1)
        #calculate autocorrelogram
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 500)
        #plt.xlim(-250, 250)
        # plot autocorrelogram
        #plt.bar(time, corr, align='center', width=1, color='black')
        #plt.plot(time, corr, '-', color='black')
        #smoothed_corr = convolve_array(corr)
        #plt.plot(time, corr, '-', color='black')
        #mean_corr = np.nanmean(np.take(corr, np.where(np.logical_and(time >= 0.5, time <=200))))
        #corr = corr - mean_corr
        f, Pxx_den = calculate_power_spectra(corr)

        plot_spectral_density(f, Pxx_den, save_path, spike_data, cluster, cluster_index)
        #each neuron was defined as θ−baselineθ+baseline, where θ is the mean power at 6–10 Hz and
        # baseline is the mean power in two adjacent frequency bands (3–5 and 11–13 Hz).

        #calculate_index
        theta_power = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f > 4, f < 12))))
        baseline = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >  0, f < 50))))
        #adjacent_power2 = np.nanmean(np.take(Pxx_den, np.where(np.logical_and(f >= 11, f <=13))))
        #baseline = (adjacent_power1 + adjacent_power2)/2
        x = theta_power - baseline
        y = theta_power + baseline
        t_index = x/y

        plot_theta_properties(time, corr, f, Pxx_den, save_path, spike_data, cluster, cluster_index, t_index)
        spike_data.at[cluster,"ThetaIndex"] = t_index

    return spike_data



def calculate_power_spectra(firing_rate):
    #f, Pxx_den = signal.welch(firing_rate, scaling='spectrum')
    f, Pxx_den = signal.periodogram(firing_rate, fs=1)
    #plt.semilogy(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()
    return f, Pxx_den


def plot_spectral_density(f, Pxx_den, save_path, spike_data, cluster, cluster_index):
    plt.semilogy(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_Spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def plot_theta_properties(time, corr, f, Pxx_den, save_path, spike_data, cluster, cluster_index, t_index):
    fig = plt.figure(figsize=(7,5)) # width, height?
    ax = fig.add_subplot(1, 2, 1)  # specify (nrows, ncols, axnum)
    ax.set_xlim(-300, 300)
    ax.plot(time, corr, '-', color='black')
    x=np.max(corr)
    ax.text(-200,x, "theta index = " + str(round(t_index,3)), fontsize =10)

    ax = fig.add_subplot(1, 2, 2)  # specify (nrows, ncols, axnum)
    ax.semilogy(f, Pxx_den)
    plt.ylim([0,20])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster_index) + '_theta_properties.png', dpi=300)
    plt.close()
    return

