#!/usr/bin/env python

""" To do:
* calibrate fluence to S/N 
* generate random FRB parameters before running, read from .txt file 
* get arrival time without argmax!

"""
import time

import random
import numpy as np
import glob
import scipy
import optparse
import random
import copy

try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass

try:
    from injectfrb import simulate_frb
    from injectfrb import reader
    from injectfrb import tools
except:
    import simulate_frb
    import reader
    import tools

k_DM=1e3/0.241

def inject_in_filterbank(fn_fil, fn_out_dir, N_FRB=1, 
                         NTIME=2**15, rfi_clean=False,
                         dm=1000.0, chunksize=2, calc_snr=True, start=0, 
                         freq_ref=None, clipping=None, 
                         upchan_factor=1, upsamp_factor=1, 
                         simulator='injectfrb', paramslist=None,
                         nbit=8):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.

    Parameters:
    -----------

    fn_fil : str
        name of filterbank file 
    fn_out_dir : str 
        directory for output files 
    N_FRB : int 
        number of FRBs to inject 
    NTIME : int 
        number of time samples per data chunk 
    rfi_clean : bool 
        apply rfi filters 
    dm : float / tuple 
        dispersion measure(s) to inject FRB with 
    dt : float 
        time resolution 
    chunksize : int 
        size of data in samples to read in 
    calc_snr : bool 
        calculates S/N of injected pulse using pulse profile 
        as matched filter
    start : int 
        start sample 
    freq_ref : float 
        reference frequency for injection code 
    clipping : 
        zero out bright events in zero-DM timestream 
    nbit : int 
        number of bits in filterbank data

    Returns:
    --------
    None 
    """
    params_arr = np.loadtxt(paramslist)
    params_arr = params_arr.transpose()
    dm_max = params_arr[0].max()
    dm_min = params_arr[0].min()
    if len(params_arr.shape)==1:
        params_arr = params_arr[:, None]

    SNRTools = tools.SNR_Tools()

    data_fil_obj_skel, freq_arr, dt, header = reader.read_fil_data(fn_fil, start=0, stop=1)
    NFREQ = header['nchans']
    BW = np.abs(header['nchans']*header['foff'])

    if freq_ref is None:
        freq_ref = 0.5*(freq_arr[0]+freq_arr[-1])

    max_dm = dm_max
    t_delay_max = abs(k_DM*max_dm*(freq_arr[0]**-2 - freq_arr[-1]**-2))
    t_delay_max_pix = int(t_delay_max / dt)

    # ensure that dispersion sweep is not too large 
    # for chunksize
    f_edge = 0.3    

    while chunksize <= t_delay_max_pix/f_edge:
        chunksize *= 2
        NTIME *= 2

    ii=0
    params_full_arr = []

    ttot = int(N_FRB*chunksize*dt)

    timestr = time.strftime("%Y%m%d-%H%M")
    fn_fil_out = '%s/%s_nfrb%d_DM%d-%d_%ssec_%s.fil' % (fn_out_dir, simulator, N_FRB, dm_min, dm_max, ttot, timestr)
    #fn_params_out = fn_fil_out.strip('.fil') + '.txt'
    fn_params_out = fn_fil_out.rsplit('.', 1)[0] + '.txt'

    print(fn_fil_out)
    print(fn_params_out)

    f_params_out = open(fn_params_out, 'w+')

    hdr_cols1 = ["DM", "Sigma", "Time", "Sample", 
                 "dfact", "W_int", "W_obs", "Spec_idx", 
                 "tau_ref", "Tsamp", "BW", 
                 "f_hi", "Nchan", "f_ref"]
    
    hdr_cols2 = ["(pc/cc)", "", "(s)", "", 
                 "", "(ms)", "(ms)", "", 
                 "", "(us)", "(MHz)", 
                 "(MHz)", "", "(MHz)"]

    hdr_fmts = ["{:^9}", "{:^8}", "{:^13}", "{:^9}", 
                "{:^5}", "{:^8}", "{:^8}", "{:^8}", 
                "{:^9}", "{:^8}", "{:^10}", 
                "{:^10}", "{:^8}", "{:^10}"]
    
    dat_fmts = ["{:8.1f}", "{:8.2f}", "{:13.4f}", "{:9d}", 
                "{:5d}", "{:8.3f}", "{:8.3f}", "{:8.2f}", 
                "{:9.3f}", "{:8.2f}", "{:10.3f}", 
                "{:10.3f}", "{:8d}", "{:10.3f}"]

    fmt_hdr = "#" + "  ".join(hdr_fmts) + "\n"
    hdr1 = fmt_hdr.format(*hdr_cols1)
    hdr2 = fmt_hdr.format(*hdr_cols2)
    f_params_out.write(hdr1)
    f_params_out.write(hdr2)
    f_params_out.close()

    fmt_out = "  ".join(dat_fmts) + "\n"
     
    print("============ HEADER INFORMATION ============")
    reader.print_filheader(header)
    kk = 0
    samplecounter = 0

    for ii in xrange(N_FRB):

        dm = params_arr[0,ii]
        fluence = params_arr[1,ii]
        width_sec = params_arr[2,ii]
        spec_ind = params_arr[3,ii]
        disp_ind = params_arr[4,ii]
        scat_tau_ref = 0.
            
        # drop FRB in random location in data chunk
        #offset = random.randint(np.int(0.1*chunksize), np.int((1-f_edge)*chunksize))
        offset = 0
        data_filobj, freq_arr, dt, header = reader.read_fil_data(fn_fil, 
                                                                  start=start+chunksize*(ii-kk), stop=chunksize)
        data = data_filobj.data            

        if ii==0:
            fn_rfi_clean = reader.write_to_fil(np.zeros([NFREQ, 0]), 
                                            header, fn_fil_out)

        # injected pulse time in seconds since start of file
        t0_ind = offset+NTIME//2+chunksize*ii 
        t0 = t0_ind*dt

        if len(data)==0:
            break             
            
        NTIME = np.int(2*t_delay_max/dt)
        if data.shape[1]<NTIME:
            print("Not enough data in the filterbank file. Not injecting.")
            return
        data_event = (data[:, offset:offset+NTIME]).astype(np.float)

        data_event, params = simulate_frb.gen_simulated_frb(NFREQ=upchan_factor*NFREQ, 
                                            NTIME=upsamp_factor*NTIME, sim=True, 
                                            fluence=fluence, spec_ind=spec_ind, width=width_sec,
                                            dm=dm, scat_tau_ref=scat_tau_ref, 
                                            background_noise=data_event, 
                                            delta_t=dt/upsamp_factor, plot_burst=False, 
                                            freq=(freq_arr[0], freq_arr[-1]), 
                                            FREQ_REF=freq_ref, scintillate=False)

        print(data_event.shape)
        data_event = data_event.reshape(NFREQ, upchan_factor, NTIME, upsamp_factor).mean(-1).mean(1)

        dm_ = params[0]
        params.append(offset)

        width_obs = np.sqrt(width_sec**2 + dt**2 + scat_tau_ref**2)
        params[2] = width_obs

        print("width_intrinsic: %0.2f\nwidth_obs: %0.2f" % (1e3*width_sec, 1e3*width_obs))
        print("%d/%d Injecting with DM:%d width_samp: %.1f offset: %d using %s" % 
                                (ii+1, N_FRB, dm_, width_obs/dt, offset, simulator))


        data[:, offset:offset+NTIME] = data_event

        width = width_obs
        downsamp = max(1, int(width/dt))
        t_delay_mid = k_DM*dm_*(freq_ref**-2-freq_arr[0]**-2)
        # this is an empirical hack. I do not know why 
        # the PRESTO arrival times are different from t0 
        # by the dispersion delay between the reference and 
        # upper frequency
        t0 -= t_delay_mid #hack

        #data_filobj.data = copy.copy(data)
        data_filobj.data = data

        # Note presto dedisperse assumes 
        if calc_snr is True:
            print("Calculating true filter")
            prof_true_filobj = copy.deepcopy(data_filobj)
            prof_true_filobj.dedisperse(dm_, ref_freq=freq_ref)
            prof_true = np.mean(prof_true_filobj.data, 0)
            prof_true = prof_true[np.where(prof_true>prof_true.max()*0.01)]
            sig_total = np.sqrt((prof_true**2).sum())
        else:
            print("not calculating")
            prof_true = None

        print(np.max(data))
        data[data>(2**nbit-1)] = 2**nbit-1
        print(np.max(data))

        data_filobj.data = copy.copy(data)
        data_filobj.dedisperse(dm_, ref_freq=freq_ref)

        start_t = abs(k_DM*dm_*(freq_arr[0]**-2 - freq_ref**-2))
        start_pix = int(start_t/dt)
        end_t = abs(k_DM*dm_*(freq_arr[-1]**-2 - freq_ref**-2))
        end_pix = int(end_t / dt)

        data_rb = data_filobj.data

        data_rb = data_rb[:, start_pix:-end_pix].mean(0)

        widths_snr = range(int(max(downsamp/2.,1)), int(min(downsamp*2, 2500)))
        print(widths_snr)
        snr_max, width_max = SNRTools.calc_snr_matchedfilter(data_rb,
                                    widths=widths_snr,
                                    true_filter=prof_true)

        #plt.plot(data_filobj.data.mean(0))
        #plt.legend(['%f %f' % (snr_max, width_max)])
        #plt.axvline(np.argmax(data_filobj.data.mean(0)), color='red')
        #plt.show()
        local_thresh = 0.
        if snr_max <= local_thresh:
            print("S/N <= %d: Not writing to file" % local_thresh)
            kk += 1
            continue
            
        print("S/N: %.2f width_used: %.1f width_tru: %.1f DM: %.1f" 
              % (snr_max, width_max, width/dt, dm_))

        # Presto dedisperses to top of band, so this is at fmax
#        t0_ind = np.argmax(data_filobj.data.mean(0)) + chunksize*ii
        t0_ind = np.argmax(data_filobj.data.mean(0)) + samplecounter
        t0 = t0_ind*dt #huge hack

        if ii<0:
            fn_rfi_clean = reader.write_to_fil(data.transpose(), header, fn_fil_out)
        elif ii>=0:
            fil_obj = reader.filterbank.FilterbankFile(fn_fil_out, mode='readwrite')
            fil_obj.append_spectra(data.transpose())

        samplecounter += data.shape[1]
        f_params_out = open(fn_params_out, 'a+')

        outstr = fmt_out.format(params[0], snr_max, t0, t0_ind, downsamp,
                                width_sec*1e3, width_obs*1e3, spec_ind,
                                scat_tau_ref, dt*1e6, BW, header['fch1'], NFREQ, freq_ref)

        print(fn_params_out)
        print(outstr)

        f_params_out.write(outstr)

        f_params_out.close()
        
        del data, data_event


if __name__=='__main__':

    def foo_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser = optparse.OptionParser(prog="inject_frb.py", \
                        version="", \
                        usage="%prog FN_FILTERBANK OUTDIR [OPTIONS]", \
                        description="Inject FRBs into filterbank data")

    parser.add_option('--sig_thresh', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 8.0)", default=8.0)

    parser.add_option('--nfrb', dest='nfrb', type='int', \
                        help="Number of FRBs to inject(Default: 50).", \
                        default=10)

    parser.add_option('--calc_snr', 
                      help="calculate S/N of injected pulse with inj signal as filter", 
                      default=False)
    

    parser.add_option('--paramslist', dest='paramslist', type='str', \
                        help="path to txt file containing FRB parameters", \
                        default=None)



    options, args = parser.parse_args()
    fn_fil = args[0]
    fn_fil_out = args[1]

    inject_in_filterbank(fn_fil, fn_fil_out, N_FRB=options.nfrb,
                         NTIME=2**15, calc_snr=False, start=0, 
                         paramslist=options.paramslist)


