import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import reader

def read_singlepulse(fn, max_rows=None, beam=None):
    """ Read in text file containing single-pulse 
    candidates. Allowed formats are:
    .singlepulse = PRESTO output
    .txt = injection pipeline output
    .trigger = AMBER output 
    .cand = heimdall output 

    max_rows sets the maximum number of 
    rows to read from textfile 
    beam is the beam number to pick in case of .trigger files
    """

    if fn.split('.')[-1] in ('singlepulse', 'txt'):
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A.shape) == 1:
            A = A[None]

        dm, sig, tt, downsample = A[:,0], A[:,1], A[:,2], A[:,4]
    elif fn.split('.')[-1] == 'trigger':
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A)==0:
            return [],[],[],[],[]

        if len(A.shape) == 1:
            A = A[None]

        # Check if amber has compacted, in which case 
        # there are two extra rows
        if len(A[0]) > 7:
            if len(A[0])==8:
                # beam batch sample integration_step compacted_integration_steps time DM compacted_DMs SNR
                beamno, dm, sig, tt, downsample = A[:, 0], A[:, -3], A[:, -1], A[:, -4], A[:, 3]
            elif len(A[0])==10:
                beamno, dm, sig, tt, downsample = A[:, 0], A[:, -3], A[:, -1], A[:, -5], A[:, 3]
            else:
                print("Error: DO NOT RECOGNIZE COLUMNS OF .trigger FILE")
                return [],[],[],[],[]
        else:
            # beam batch sample integration_step time DM SNR
            beamno, dm, sig, tt, downsample = A[:, 0], A[:,-2], A[:,-1], A[:, -3], A[:, 3]
        
        if beam is not None and beam != 'all':
                # pick only the specified beam
                dm = dm[beamno.astype(int) == beam]
                sig = sig[beamno.astype(int) == beam]
                tt = tt[beamno.astype(int) == beam]
                downsample = downsample[beamno.astype(int) == beam]

    elif fn.split('.')[-1] == 'cand':
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A.shape) == 1:
            A = A[None]
        
        # SNR sample_no time log_2_width DM_trial DM Members first_samp last_samp
        #dm, sig, tt, downsample = A[:,5], A[:,0], A[:, -5], A[:, -4]
        dm, sig, tt, downsample = A[:,6], A[:,0], A[:, -5], A[:, -4]

        try:
            beamno = A[:, 9]
            return dm, sig, tt, downsample, beamno
        except:
            pass
    elif fn.split('.')[-1]=='fredda':
        A = np.genfromtxt(fn, max_rows=max_rows)

        if len(A.shape)==1:
            A = A[None]
        
        dm, sig, tt, downsample = A[:,5], A[:,0], A[:, 2], A[:, 3]
    else:
        print("Didn't recognize singlepulse file")
        return [],[],[],[],[]

    if len(A) == 0:
        if beam == 'all':
            return 0, 0, 0, 0, 0
        else:
            return 0, 0, 0, 0

    if beam == 'all':
        return dm, sig, tt, downsample, beamno
    else:
        return dm, sig, tt, downsample

def dm_range(dm_max, dm_min=5., frac=0.2):
    """ Generate list of DM-windows in which 
    to search for single pulse groups. 

    Parameters
    ----------
    dm_max : float 
        max DM 
    dm_min : float  
        min DM 
    frac : float 
        fractional size of each window 

    Returns
    -------
    dm_list : list 
        list of tuples containing (min, max) of each 
        DM window
    """

    dm_list =[]
    prefac = (1-frac)/(1+frac)

    while dm_max>dm_min:
        if dm_max < 100.:
            prefac = (1-2*frac)/(1+2*frac)
        if dm_max < 50.:
            prefac = 0.0 

        dm_list.append((int(prefac*dm_max), int(dm_max)))
        dm_max = int(prefac*dm_max)

    return dm_list


def get_triggers(fn, sig_thresh=5.0, dm_min=0, dm_max=np.inf, 
                 t_window=0.5, max_rows=None, t_max=np.inf,
                 sig_max=np.inf, dt=2*40.96, delta_nu_MHz=300./1536, 
                 nu_GHz=1.4, fnout=False, tab=None, read_beam=False, 
                 dm_width_filter=False, return_clustcounts=False):
    """ Get brightest trigger in each 10s chunk.

    Parameters
    ----------
    fn : str 
        filename with triggers (.npy, .singlepulse, .trigger)
    sig_thresh : float
        min S/N to include
    dm_min : 
        minimum dispersion measure to allow 
    dm_max : 
        maximum dispersion measure to allow 
    t_window : float 
        Size of each time window in seconds
    max_rows : 
        Only read this many rows from raw trigger file 
    fnout : str 
        name of text file to save clustered triggers to 
    tab : int
        which TAB to process (0 for IAB)
    read_beam: bool
        read and return beam number (default: False)
        all beams are read if this is true
    return_clustcounts : bool 
        return array of number of candidates per 
        trigger cluster

    Returns
    -------
    sig_cut : ndarray
        S/N array of brightest trigger in each DM/T window 
    dm_cut : ndarray
        DMs of brightest trigger in each DM/T window 
    tt_cut : ndarray
        Arrival times of brightest trigger in each DM/T window 
    ds_cut : ndarray 
        downsample factor array of brightest trigger in each DM/T window
    beam_cut: ndarray
        beam array of brightest trigger in each DM/T windows (only if read_beam is True)
    ind_cut : ndarray
        indexes of events that were kept 
    """
    if tab is not None:
        beam_amber = tab
        read_beam = False
    elif read_beam:
        beam_amber = 'all'
    else:
        beam_amber = None

    if type(fn) == str:
        if read_beam:
            dm, sig, tt, downsample, beam = read_singlepulse(fn, max_rows=max_rows, beam=beam_amber)[:5]
        else:
            dm, sig, tt, downsample = read_singlepulse(fn, max_rows=max_rows, beam=beam_amber)[:4]
    elif type(fn) == np.ndarray:
        dm, sig, tt, downsample = fn[:, 0], fn[:, 1], fn[:, 2], fn[:, 3]
    else:
        print("Wrong input type. Expected string or ndarray")
        if read_beam:
            return [], [], [], [], [], []
        else:
            return [], [], [], [], []

    ntrig_orig = len(dm)

    bad_sig_ind = np.where((sig < sig_thresh) | (sig > sig_max))[0]
    sig = np.delete(sig, bad_sig_ind)
    tt = np.delete(tt, bad_sig_ind)
    dm = np.delete(dm, bad_sig_ind)
    downsample = np.delete(downsample, bad_sig_ind)
    sig_cut, dm_cut, tt_cut, ds_cut = [], [], [], []
    ntrig_clust_arr = []

    if read_beam:
        beam = np.delete(beam, bad_sig_ind)
        beam_cut = []

    if len(tt) == 0:
        print("Returning None: time array is empty")
        return 

    tduration = tt.max() - tt.min()
    ntime = int(tduration / t_window)

    # Make dm windows between 90% of the lowest trigger and 
    # 10% of the largest trigger
    if dm_min == 0:
        dm_min = 0.9*dm.min()
    if dm_max > 1.1*dm.max():
        dm_max = 1.1*dm.max()

    # Can either do the DM selection here, or after the loop
#    dm_list = dm_range(dm_max, dm_min=dm_min)
    dm_list = dm_range(1.1*dm.max(), dm_min=0.9*dm.min())

    print("\nGrouping in window of %.2f sec" % np.round(t_window,2))
    print("DMs:", dm_list)

    tt_start = tt.min() - .5*t_window
    ind_full = []

    # might wanna make this a search in (dm,t,width) cubes
    for dms in dm_list:
        for ii in xrange(ntime + 2):
            try:    
                # step through windows of t_window seconds, starting from tt.min()
                # and find max S/N trigger in each DM/time box
                t0, tm = t_window*ii + tt_start, t_window*(ii+1) + tt_start
                ind = np.where((dm<dms[1]) & (dm>dms[0]) & (tt<tm) & (tt>t0))[0] 
                ntrig_clust = len(ind)

                if ntrig_clust==0:
                    continue
                else:
                    ntrig_clust_arr.append(ntrig_clust)
                    
                ind_maxsnr = ind[np.argmax(sig[ind])]
                sig_cut.append(sig[ind_maxsnr])
                dm_cut.append(dm[ind_maxsnr])
                tt_cut.append(tt[ind_maxsnr])
                ds_cut.append(downsample[ind_maxsnr])
                if read_beam:
                    beam_cut.append(beam[ind_maxsnr])
                ind_full.append(ind_maxsnr)
            except:
                continue

    ind_full = np.array(ind_full)
    dm_cut = np.array(dm_cut)
    # now remove the low DM candidates
    tt_cut = np.array(tt_cut).astype(np.float)
    ind = np.where((dm_cut >= dm_min) & (dm_cut <= dm_max) & (tt_cut < t_max))[0]

    dm_cut = dm_cut[ind]
    ind_full = ind_full[ind]
    sig_cut = np.array(sig_cut)[ind]
    tt_cut = tt_cut[ind]
    ds_cut = np.array(ds_cut)[ind]
    ntrig_clust_arr = np.array(ntrig_clust_arr)[ind]
    if read_beam:
        beam_cut = np.array(beam_cut)[ind]

    ntrig_group = len(dm_cut)

    print("Grouped down to %d triggers from %d\n" % (ntrig_group, ntrig_orig))

    rm_ii = []

    if dm_width_filter:
        for ii in xrange(len(ds_cut)):        
            tdm = 8.3 * delta_nu_MHz / nu_GHz**3 * dm_cut[ii] # microseconds#

            if ds_cut[ii]*dt < (0.5*(dt**2 + tdm**2)**0.5):
                rm_ii.append(ii)

    dm_cut = np.delete(dm_cut, rm_ii)
    tt_cut = np.delete(tt_cut, rm_ii)
    sig_cut = np.delete(sig_cut, rm_ii)
    ds_cut = np.delete(ds_cut, rm_ii)
    ntrig_clust_arr = np.delete(ntrig_clust_arr, rm_ii)
    if read_beam:
        beam_cut = np.delete(beam_cut, rm_ii)
    ind_full = np.delete(ind_full, rm_ii)

    if fnout:
        if read_beam:
            clustered_arr = np.concatenate([sig_cut, dm_cut, tt_cut, ds_cut, beam_cut, ind_full])
            clustered_arr = clustered_arr.reshape(6, -1)
        else:
            clustered_arr = np.concatenate([sig_cut, dm_cut, tt_cut, ds_cut, ind_full])
            clustered_arr = clustered_arr.reshape(5, -1)
        np.savetxt(fnout, clustered_arr) 

    if not return_clustcounts:
        if read_beam:
            return sig_cut, dm_cut, tt_cut, ds_cut, beam_cut, ind_full
        else:
            return sig_cut, dm_cut, tt_cut, ds_cut, ind_full
    elif return_clustcounts:
        if read_beam:
            return sig_cut, dm_cut, tt_cut, ds_cut, beam_cut, ind_full, ntrig_clust_arr
        else:
            return sig_cut, dm_cut, tt_cut, ds_cut, ind_full, ntrig_clust_arr

def plotfour(dataft, datats, datadmt, header,
             beam_time_arr=None, figname_out=None, dm=0,
             dms=[0,1], 
             datadm0=None, suptitle='', heimsnr=-1,
             ibox=1, ibeam=-1, prob=-1,
             showplot=True,multibeam_dm0ts=None,
             fnT2clust=None,imjd=0.0,fake=False):
    """ Plot a trigger's dynamics spectrum, 
        dm/time array, pulse profile, 
        multibeam info (optional), and zerodm (optional)
        Parameter
        ---------
        dataft : 
            freq/time array (nfreq, ntime)
        datats : 
            dedispersed timestream
        datadmt : 
            dm/time array (ndm, ntime)
        beam_time_arr : 
            beam time SNR array (nbeam, ntime)
        figname_out : 
            save figure with this file name 
        dm : 
            dispersion measure of trigger 
        dms : 
            min and max dm for dm/time array 
        datadm0 : 
            raw data timestream without dedispersion
    """

    classification_dict = {'prob' : [],
                           'snr_dm0_ibeam' : [],
                           'snr_dm0_allbeam' : []}
    datats /= np.std(datats[datats!=np.max(datats)])
    datats -= np.median(datats[datats!=np.max(datats)])
    nfreq, ntime = dataft.data.shape
    xminplot,xmaxplot = 500.-300*ibox/16.,500.+300*ibox/16 # milliseconds
    if xminplot<0:
        xmaxplot=xminplot+500+300*ibox/16        
        xminplot=0
    xminplot,xmaxplot = 0, 1000.
    dm_min, dm_max = dms[0], dms[1]
    tmin, tmax = 0., 1e3*dataft.dt*ntime
    freqmax = header['fch1']
    freqmin = freqmax + header['nchans']*header['foff']
    freqs = np.linspace(freqmin, freqmax, nfreq)
    tarr = np.linspace(tmin, tmax, ntime)
#    fig = plt.figure(figsize=(8,10))
    fig, axs = plt.subplots(2, 2, figsize=(8,10), constrained_layout=True)

    if fake:
        fig.patch.set_facecolor('red')
        fig.patch.set_alpha(0.5)

    extentft=[tmin,tmax,freqmin,freqmax]
    axs[0][0].imshow(dataft.data, aspect='auto',extent=extentft, interpolation='nearest')
    DM0_delays = xminplot + dm * 4.15E6 * (freqmin**-2 - freqs**-2)
    axs[0][0].plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    axs[0][0].set_xlim(xminplot,xmaxplot)
    axs[0][0].set_xlabel('Time (ms)')
    axs[0][0].set_ylabel('Freq (MHz)')
    if prob!=-1:
        axs[0][0].text(xminplot+50*ibox/16.,0.5*(freqmax+freqmin),
                       "Prob=%0.2f" % prob, color='white', fontweight='bold')
        classification_dict['prob'] = prob

#    plt.subplot(322)
    extentdm=[tmin, tmax, dm_min, dm_max]
    axs[0][1].imshow(datadmt[::-1], aspect='auto',extent=extentdm)
    axs[0][1].set_xlim(xminplot,xmaxplot)
    axs[0][1].set_xlabel('Time (ms)')
    axs[0][1].set_ylabel(r'DM (pc cm$^{-3}$)')

#    plt.subplot(323)
    axs[1][0].plot(tarr, datats)
    axs[1][0].grid('on', alpha=0.25)
    axs[1][0].set_xlabel('Time (ms)')
    axs[1][0].set_ylabel(r'Power ($\sigma$)')
    axs[1][0].set_xlim(xminplot,xmaxplot)
    axs[1][0].text(0.51*(xminplot+xmaxplot), 0.5*(max(datats)+np.median(datats)), 
            'Heimdall S/N : %0.1f\nHeimdall DM : %d\
            \nHeimdall ibox : %d\nibeam : %d' % (heimsnr,dm,ibox,ibeam), 
            fontsize=8, verticalalignment='center')
    
    fig.suptitle(suptitle, color='C1')
    if fake:
        fig.suptitle('INJECTION')

    if figname_out is not None:
        fig.savefig(figname_out)
    if showplot:
        fig.show()

    return 


def dm_transform(data, dm_max=20,
                 dm_min=0, dm0=None, ndm=64, 
                 freq_ref=None, downsample=16):
    """ Transform freq/time data to dm/time data.                                                                                                                                           
    """
    ntime = int(data.data.shape[1])
    dm_min = max(0, dm_min)

    dms = np.linspace(dm_min, dm_max, ndm, endpoint=True)

    # if dm0 is not None:
    #     dm_max_jj = np.argmin(abs(dms-dm0))
    #     dms += (dm0-dms[dm_max_jj])

    data_full = np.zeros([ndm, ntime//downsample])

    for ii, dm in enumerate(dms):
        data.dedisperse(dm)
        _dts = np.mean(data.data,axis=0)
        data_full[ii] = _dts[:ntime//downsample*downsample].reshape(ntime//downsample, downsample).mean(1)

    return data_full, dms

def make_candplots(fnfil, fncand, ndm=32):
    dm, sig, tt, downsample = read_singlepulse(fncand)
    ncand = len(dm)
    _, freq, dt, header = reader.read_fil_data(fnfil, start=0, stop=1)

    for ii in range(ncand):
        dm0, sig0, tt0, downsample0 = dm[ii], sig[ii], tt[ii], downsample[ii]
        downsample0 = int(downsample0)
        disp_delay = 4140*dm0*(np.abs(freq[0]**-2 - freq[-1]**-2))
        start_ii = int((tt0-0.5)/dt)
        stop_ii = 16384
        data, freq, dt, header = reader.read_fil_data(fnfil, start=start_ii, stop=stop_ii)
        datadm, dms = dm_transform(data, dm_max=dm0+50,
                               dm_min=dm0-50, dm0=dm0, ndm=ndm, 
                               freq_ref=np.mean(freq), 
                               downsample=downsample0)
        data.dedisperse(dm0)
        data.downsample(downsample0)
        plotfour(data, data.data.mean(0), datadm, header,
                dm=dm0, ibox=downsample0, dms=dms,
                heimsnr=sig0, 
                showplot=False, figname_out='outfig%d.png'%ii)

def plot_cluster(fn, cluster_index, figname_out):
    dm, sig, tt, downsample = read_singlepulse(fn, max_rows=None, beam=None)
    fig = plt.figure(figsize=(12,6))
    plt.scatter(tt, dm, c=downsample, alpha=0.25, s=sig, cmap='RdBu')
    plt.scatter(tt[cluster_index], dm[cluster_index], color='k', s=5)
    plt.savefig(figname_out)

if __name__=='__main__':
    t_window = 5.0
    fn = sys.argv[1]
    fnout = sys.argv[2]
    fnfil = sys.argv[3]
    giants_raw = np.genfromtxt(fn)
    sig_cut, dm_cut, tt_cut, ds_cut, cluster_index = get_triggers(fn, t_window=t_window)
    plot_cluster(fn, cluster_index, 'cluster.png')
    fmt = '%0.5f','%d','%d','%0.3f','%d','%d','%0.2f','%d'
    np.savetxt(fnout, giants_raw[cluster_index], fmt=fmt)
    #make_candplots(fnfil, fnout)

