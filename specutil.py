import math
from collections import Counter
import numpy as np
from scipy import signal

def stft(X, nfft=1024, noverlap=None, window='hann', fs=1.0, S=0, G=0, Gr=0):
    """Computes the short-time Discrete Fourier Transform of X.

    This function doesn't pad or extend the input but truncates it
    to fit into an integer number of windows (with overlap). It also
    renormalizes values to correspond to amplitudes in the original
    signal.

    Note: the definition of decibels used below corresponds to
        x dB = 20 * log10(x)

    Args:
        X (numpy.ndarray): The uniformly sampled signal to perform STFT
            on. If it has more than one axis, it is flattened; if it
            has imaginary parts, those are discarded.
        nfft (int): The number of points to use in one FFT pass (for
            one segment), defaults to 1024, must be positive. If it is
            larger than the length of ``X`` (flattened), all points are
            used in a single pass.
        noverlap (int): The number of overlapping points to use in
            consecutive segments, defaults to nfft // 2, must be
            positive and smaller than nfft.
        window (str or tuple): The type of window to use. This is
            passed to `scipy.signal.get_window`, see the docstring of
            `get_window` for available values. This function uses
            a DFT-even (periodic) window with ```nfft``` points.
        fs (float): sample rate of the signal, used to compute output
            times, defaults to 1.0.
        S (float): sensitivity of the instrument used to produce the
            original signal in dB output units / input units. Defaults
            to 0.
        G (float): preamplifier gain in dB. Defaults to 0 (no gain).
        Gr (float): gain of the A/D converter used to sample ``X`` in dB
            output units (units of ``X``) / input units. Defaults to 0.

    Returns:
        spec_mag (numpy.ndarray): normalized magnitude of the complex
            DFT values with shape ``(freq_bins, n_samples)``.
        spec_phase (numpy.ndarray): phase angle of the complex DFT
            values with shape ``(freq_bins, n_samples)``.
        freqs (numpy.ndarray): DFT frequency bins.
        times (numpy.ndarray): time values computed from sample rate.

    Raises:
        TypeError: if any arguments are of the wrong type.
        ValueError: if any arguments are outside the accepted range.
    """

    # Check and adjust input.
    _check_stft_input(X, nfft, noverlap, window, fs, S, G, Gr)
    X = X.reshape(-1,1).real # Reshape to column vector.
    nfft = int(nfft)
    noverlap = int(noverlap) if noverlap is not None else nfft // 2
    if X.shape[0] <= nfft: # Use all points.
        nfft = X.shape[0]
        noverlap = 0

    # Truncate X to fit the windows.
    step = nfft - noverlap
    nsegments = (X.shape[0] - nfft) // step + 1
    overflow = (X.shape[0] - nfft) % step
    X = X[:-overflow,:] if overflow != 0 else X

    # Create window.
    f_window = signal.get_window(window, nfft, fftbins=True)
    f_window /= np.sum(f_window) # Normalize to have real amplitudes.
    f_window = f_window.reshape(-1,1) # Reshape to column vector.

    # Compute fft for each segment.
    X_segments = (np.hstack((X[i*step:i*step+nfft,:]
        for i in range(nsegments))) if nsegments > 1 else X) * f_window
    spectrum = np.fft.rfft(X_segments, n=nfft, axis=0, norm=None)

    # Separate complex parts and adjust magnitude.
    spec_mag = np.abs(spectrum)
    spec_phase = np.angle(spectrum)
    spec_mag[1:] *= 2 # Compensate for energy lost to negative frequencies.
    spec_mag *= 10 ** (- (S + G + Gr) / 20) # Adjust with parameters.

    # Calculate frequencies and times.
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    times = np.arange(nsegments) * step / fs

    return spec_mag, spec_phase, freqs, times

def _check_stft_input(X, nfft, noverlap, window, fs, S, G, Gr):
    """Sanitizes the input to stft."""
    # X
    if type(X) != np.ndarray:
        raise TypeError("Input X is not a numpy array.")
    if np.prod(X.shape) == 0:
        raise ValueError("Input X is empty.")

    # nfft
    try:
        int(nfft)
    except TypeError:
        raise TypeError("nfft is not of an integer type.")
    if int(nfft) != nfft:
        raise TypeError("nfft is not an integer.")
    if nfft <= 0:
        raise ValueError("nfft must be positive.")

    # noverlap
    try:
        int(noverlap)
    except TypeError:
        raise TypeError("noverlap is not of an integer type.")
    if int(noverlap) != noverlap:
        raise TypeError("noverlap is not an integer.")
    if nfft <= 0:
        raise ValueError("noverlap must be positive.")
    if nfft <= noverlap:
        raise ValueError("noverlap must be smaller than nfft.")

    # window
    try:
        signal.get_window(window, 1)
    except ValueError:
        print("Unkown window " + str(window))

    # fs
    try:
        complex(fs)
    except TypeError:
        raise TypeError("fs is not a number.")
    if fs != float(complex(fs).real):
        raise TypeError("fs is not a real number.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    # S
    try:
        complex(S)
    except TypeError:
        raise TypeError("S is not a number.")
    if S != float(complex(S).real):
        raise TypeError("S is not a real number.")

    # G
    try:
        complex(G)
    except TypeError:
        raise TypeError("G is not a number.")
    if G != float(complex(G).real):
        raise TypeError("G is not a real number.")

    # Gr
    try:
        complex(Gr)
    except TypeError:
        raise TypeError("Gr is not a number.")
    if Gr != float(complex(Gr).real):
        raise TypeError("Gr is not a real number.")

def label(samples, spectrum, iv_lists, nfft=4096, noverlap=2048, fs=44100):
    """Generates input values and target classes for classification.

    The spectrogram is split into slices, the wave samples are split
    into packets containing as many points as were used to compute
    the spectra.
    """
    step = nfft - noverlap
    # Helper functions.
    flatten = lambda l: ([l[0]] if not isinstance(l[0], list) else \
              flatten(l[0])) + flatten(l[1:]) if len(l) > 0 else l
    transform = lambda iv: (int(round(iv[0] * 60 * fs)),
                            int(round(iv[1] * 60 * fs))) # Minutes ==> samples.
    wframe = lambda start, end: (math.ceil(start / step) * step,
                                 math.floor(end / step) * step)
    sframe = lambda start, end: (math.ceil(start / step),
                                 math.ceil((end - nfft) / step))
    sframediff = lambda start, end: math.ceil((end - nfft) / step) - math.ceil(start / step)
    segment = lambda X: np.hstack(X[i*step:i*step+nfft,np.newaxis] for i in range((X.shape[0]-nfft)//step+1)) \
                        if X.shape[0] >= nfft else np.empty((1,1))
    # Create labeled list of intervals.
    iv_classes = flatten([[(transform(iv), i) for iv in iv_lists[i]] for i in range(len(iv_lists))])
    # Segment input.
    sample_values = np.hstack(segment(samples[slice(*wframe(start, end))]) for (start, end), _ in iv_classes)
    spectrum_values = np.hstack(spectrum[:,slice(*sframe(start, end))] for (start, end), _ in iv_classes)
    # Generate target labels.
    classes = np.hstack(np.tile(class_, (1, sframediff(start, end))) for (start, end), class_ in iv_classes)
    return sample_values.T, spectrum_values.T, classes.T # Keras convention.

def label_spec_parts(spectrum, iv_lists, nfft=4096, noverlap=2048, fs=44100):
    """Generates input values and target classes for classification.

    The spectrogram is split into slices, according to the intervals
    in iv_lists - as (start minute, end minute) tuples -, and a label
    is assigned to each slice, using increasing numbers for the classes.
    """
    step = nfft - noverlap
    # Helper functions.
    flatten = lambda l: ([l[0]] if not isinstance(l[0], list) else \
              flatten(l[0])) + flatten(l[1:]) if len(l) > 0 else l
    transform = lambda iv: (int(round(iv[0] * 60 * fs)),
                            int(round(iv[1] * 60 * fs))) # Minutes ==> samples.
    frame = lambda iv: (math.ceil(iv[0] / step), math.ceil((iv[1] - nfft) / step)) # Samples ==> FFT frames.
    # Segment input.
    slices = np.array([spectrum[:,slice(*frame(transform(iv)))].T for iv in flatten(iv_lists)],
                      dtype=object)
    # Generate target labels.
    classes = np.array(flatten([[i for iv in iv_lists[i]] for i in range(len(iv_lists))]))
    return slices, classes

def split_spec(slices, slice_labels, length):
    """Transforms spectrogram slices into a
    continuous 2D input of shorter segments.

    It takes a list of (freqs,timesteps)-shaped slices
    and a list of corresponding labels (like the output
    of label_spec_parts). Returns a 3D numpy array.
    """
    split_slices = []
    split_labels = []
    for i in range(len(slices)):
        slice_ = slices[i]
        parts = [slice_[k*length:(k+1)*length,:] for k in range(slice_.shape[0] // length)]
        split_slices.extend(parts)
        split_labels.append(np.tile(slice_labels[i], len(parts)))
    # Keras convention: (samples, timesteps, channels/features:frequencies)
    values = np.stack(split_slices, axis=0)
    labels = np.concatenate(split_labels, axis=0)
    return values, labels

def equalize_classes(values, labels):
    """Discards a minimum number of samples to achieve
    an equal class ratio.

    Samples are discarded from the end of the sequence.
    """
    orig_counts = Counter(labels)
    max_count = min(orig_counts.values())
    n_classes =  len(orig_counts.keys())
    class_counter = Counter()
    new_values = np.empty((max_count * n_classes,) + values.shape[1:], dtype=values.dtype)
    new_labels = np.empty((max_count * n_classes,), dtype=labels.dtype)
    i = 0
    for label, elt in zip(labels, values):
        if label not in class_counter or class_counter[label] < max_count:
            new_values[i] = elt
            new_labels[i] = label
            class_counter[label] += 1
            i += 1
    return new_values, new_labels

def flatten(sequence, ltypes=(list,)):
    output = []
    stack = []
    i = 0
    while i < len(sequence) or len(stack) > 0:
        if i == len(sequence):
            sequence, i = stack.pop()
            continue
        elt = sequence[i]
        i += 1
        if isinstance(elt, ltypes):
            stack.append((sequence, i))
            sequence = elt
            i = 0
            continue
        output.append(elt)
    return output

def iter_flatten(iterable, ltypes=(list,)):
    for elt in iterable:
        if isinstance(elt, ltypes):
            yield from iter_flatten(elt, ltypes)
        else:
            yield elt
