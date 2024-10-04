import numpy as np
from scipy.io.wavfile import write
import IPython.display as ipd

def linear_interpolation(bits, carrier_freq, sample_rate, duration, transition_duration=0.005):
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    bit_duration = num_samples // len(bits)

    # Create modulated signal
    modulated_signal = np.zeros(num_samples, dtype=complex)
    
    # Number of samples for smooth transition
    transition_samples = int(sample_rate * transition_duration)

    for i, bit in enumerate(bits):
        bit_value = -1.0 if bit == 0 else 1.0
        start = i * bit_duration
        end = (i + 1) * bit_duration

        # Linear interpolation for smooth transition at the beginning
        if i > 0:
            previous_value = -1.0 if bits[i - 1] == 0 else 1.0
            for j in range(transition_samples):
                alpha = j / transition_samples
                smooth_value = (1 - alpha) * previous_value + alpha * bit_value
                modulated_signal[start - transition_samples + j] = smooth_value * np.exp(1j * 2 * np.pi * carrier_freq * t[start - transition_samples + j])

        # Main bit duration signal
        modulated_signal[start:end] = bit_value * np.exp(1j * 2 * np.pi * carrier_freq * t[start:end])

        # Linear interpolation for smooth transition at the end
        if i < len(bits) - 1:
            next_value = -1.0 if bits[i + 1] == 0 else 1.0
            for j in range(transition_samples):
                alpha = j / transition_samples
                smooth_value = (1 - alpha) * bit_value + alpha * next_value
                modulated_signal[end - transition_samples + j] = smooth_value * np.exp(1j * 2 * np.pi * carrier_freq * t[end - transition_samples + j])

    return modulated_signal

def main(bits, carrier_freq=19000, sample_rate=48000, duration=5):
    # Generate BPSK modulated signal with linear interpolation smoothing
    modulated_signal_complex = linear_interpolation(bits, carrier_freq, sample_rate, duration)

    # Convert to real part
    modulated_signal = np.real(modulated_signal_complex)

    # Normalize to range [-1, 1]
    modulated_signal /= np.max(np.abs(modulated_signal))

    # Save to WAV file
    wav_filename = '/content/bpsk_signal_linear_interp.wav'
    write(wav_filename, sample_rate, (modulated_signal * 32767).astype(np.int16))

    # Display the WAV file
    return wav_filename

# Example bit sequence
bits = [0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1]

wav_filename = main(bits)
