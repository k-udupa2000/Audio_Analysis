from helper_functions import *

def Segment(filename, plotting = False, write_audio = False):
    sig, samplerate = open_audio(filename)
    sig = sig[:, 0] 
    duration = len(sig)/samplerate
    time = np.arange(0, duration, 1/samplerate)

    analytic_signal = hilbert(sig)
    amp_envelope = np.abs(analytic_signal)
    avg = sig[0]
    lag_weight = 0.9999
    cont_amp = [avg]
    for i in range(1, len(amp_envelope)):
        avg = (avg*lag_weight + amp_envelope[i]*(1 - lag_weight))
        cont_amp.append(avg)
    smooth_envelope = low_pass_filter(cont_amp, 0.0005)

    expo_envelope = np.exp(smooth_envelope + 1)

    smooth = gaussian_filter1d(expo_envelope, 100)
    smooth_d2 = np.gradient(np.gradient(smooth))
    smooth_d2 = smooth_d2/max(smooth_d2)
    infls = np.where(np.diff((smooth_d2) > 0.1))[0]

    scaled_infls = infls/samplerate
    groupedInflex = []
    ind = 0
    while ind < len(infls):
        l = getNearPoints(infls, ind, len(infls), samplerate)
        ind += len(l)
        groupedInflex.append(l)

    final_points = []
    final_points_scaled = []
    for l in groupedInflex:
        if len(l) == 1:
            mid = l[0]
        else:
            mid = (int((l[0] + l[1])/2))
        final_points.append(mid)
        final_points_scaled.append(mid/samplerate)
    list_of_tones = get_tone_arrays(final_points, infls, samplerate, sig)
    
    if plotting:
        plt.show()
        # plt.plot(time, sig)
        # plt.show()
        plt.plot(time, amp_envelope)
        plt.plot(time, cont_amp)
        plt.xlabel('time')
        plt.ylabel('s(t)')
        plt.show()
        plt.plot(time, expo_envelope)
        for i, infl in enumerate(scaled_infls, 1):
            plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
            plt.legend(bbox_to_anchor=(1.55, 1.0))
        plt.plot(time, sig)
        plt.show()
        colors = ['red', 'green', 'blue', 'orange', 'yellow', 'grey', 'indigo']
        for i, infl in enumerate(final_points_scaled, 1):
            plt.axvline(x=infl, color='b', label=f'Inflection Point {i}')
        plt.legend(bbox_to_anchor=(1.55, 1.0))
        plt.plot(time, sig)
        col_ind = 0
        for i in range(len(final_points_scaled) - 1):
            plt.fill_between(time, expo_envelope, -1, where= (time >= final_points_scaled[i]) & (time <= final_points_scaled[i + 1]), color = colors[col_ind])
            col_ind += 1
            if col_ind == len(colors):
                col_ind = 0
        plt.fill_between(time, expo_envelope, -1, where= (time >= final_points_scaled[len(final_points_scaled) - 1]) & (time <= max(time)), color = colors[col_ind])
        plt.gca().set_ylim(bottom = 0)
        plt.show()
        plt.plot(time, cont_amp)
        plt.plot(time, smooth_envelope)
        plt.show()
        plt.plot(time, sig)
        for i, infl in enumerate(final_points_scaled, 1):
            plt.axvline(x=infl, color='g', label=f'Inflection Point {i}')
        plt.legend(bbox_to_anchor=(1.55, 1.0))
        plt.legend(['Mean curve', 'Audio Split Points'])
        plt.xlabel('time')
        plt.ylabel('s(t)')
        plt.show()
    if write_audio:
        writeToAudioFile(list_of_tones, samplerate)
    return list_of_tones