import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
import pickle



# Load in the Compressed Files
with open(r"/Main_Compressed_files/filtered_VTF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_VHF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_VLF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_UTF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_UHF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_ULF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_STF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_SHF.pkl", 'rb') as f:
with open(r"/Main_Compressed_files/filtered_SLF.pkl", 'rb') as f:
    data = pickle.load(f)

#Extract the list of arrays
filtered_indices = data['indices']
filtered_real_parts = data['real_parts']
filtered_imag_parts = data['imag_parts']
print("Filtered data loaded successfully.")

N = 25920
start_time = pd.to_datetime('2014-09-16 00:00')
t = pd.date_range(start=start_time, periods=N, freq='5min')
t_array = t.to_numpy()

# Initialize an array to hold the reconstructed signals
reconstructed_signals = np.zeros((137, N))

# Reconstruct the signal for each point
for idx in range(137):
# Initialize an array to hold the spectrum
    spectrum = np.zeros(N, dtype=complex)

# Reconstruct the filtered spectrum
    indices = np.array(filtered_indices[idx])
    real_parts = np.array(filtered_real_parts[idx])
    imag_parts = np.array(filtered_imag_parts[idx])
    spectrum[indices] = real_parts + 1j * imag_parts
#Ensure there is Hermitian symmetry to confirm that the reconstructed signal will be a real signal
    spectrum[-indices] = np.conj(spectrum[indices])

# Perform the IFFT
    reconstructed_signals[idx, :] = ifft(spectrum).real

#Save the reconstructed signal(s)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_V_tidal.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_V_highfreq.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_V_lowfreq.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_U_tidal.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_U_highfreq.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_U_lowfreq.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_S_tidal.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_S_highfreq.npy", reconstructed_signals)
np.save(r"/reconstructed_arrays/Main_reconstructed_signal_S_lowfreq.npy", reconstructed_signals)

#Load the reconstructed component signals, plus the mean of the original U, V, and SE
reconstructed_signal_T = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_S_tidal.npy")
reconstructed_signal_high = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_S_highfreq.npy")
reconstructed_signal_low = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_S_lowfreq.npy")
mean_values = np.load(r"/Main_Channel_Data_Arrays/MainSE_mean_values.npy")

reconstructed_signal_T = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_U_tidal.npy")
reconstructed_signal_high = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_U_highfreq.npy")
reconstructed_signal_low = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_U_lowfreq.npy")
mean_values = np.load(r"/Main_Channel_Data_Arrays/MainU_mean_values.npy")

reconstructed_signal_T = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_V_tidal.npy")
reconstructed_signal_high = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_V_highfreq.npy")
reconstructed_signal_low = np.load(r"/reconstructed_arrays/Main_reconstructed_signal_V_lowfreq.npy")
mean_values = np.load(r"/Main_Channel_Data_Arrays/MainV_mean_values.npy")

#Initialize an array to store the fully reconstructed signal
reconstruction = np.zeros_like(reconstructed_signals)

#Add the mean, low frequency, high frequency and tidal frequency signals together to recreate the original signal for every point
for i in range(reconstructed_signals.shape[0]):
    z = mean_values[i]
    reconstruction[i] = (z + reconstructed_signal_T[i] + reconstructed_signal_low[i] + reconstructed_signal_high[i])

Print(reconstructed_signals.shape)