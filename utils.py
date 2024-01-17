import numpy as np
def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = nextpow2(sig.shape[0] + refsig.shape[0] +1)

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


def normalize(input_tensor):
    min_val, max_val = np.min(input_tensor), np.max(input_tensor)
    return -1 +2*((input_tensor-min_val)/(max_val-min_val))

def normalize_01(input_tensor):
    min_val, max_val = np.min(input_tensor), np.max(input_tensor)
    return ((input_tensor-min_val)/(max_val-min_val))

def denormalize(input_tensor, min_src, max_src):
    input_tensor = (input_tensor+1)/2
    input_tensor = input_tensor * (max_src-min_src) + min_src
    return input_tensor


def compute_correlation_time(signal):
    # Compute the autocorrelation function

    autocorr = np.fft.ifftshift(np.fft.ifft(np.fft.fft(signal)*np.conj(np.fft.fft(signal))))

    # Normalize the autocorrelation function
    normalized_autocorr = autocorr / np.max(autocorr)

    # Find the -3 dB points
    half_max = 0.5
    indices = np.where(normalized_autocorr >= half_max)[0]
    left_3db, right_3db = indices[0], indices[-1]

    # Compute the correlation time as the width between the -3 dB points
    correlation_time_samples = right_3db - left_3db

    return correlation_time_samples

def add_white_gaussian_noise(signal, snr_dB, ref_mic=0):
    # Calculate the signal power
    signal_power = np.mean(np.abs(signal[ref_mic]) ** 2)

    # Calculate the noise power using the SNR
    noise_power = signal_power / (10 ** (snr_dB / 10))

    # Generate white Gaussian noise
    noise = np.sqrt(noise_power) * np.random.randn(signal.shape[0], signal.shape[1])

    # Add the noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, noise
