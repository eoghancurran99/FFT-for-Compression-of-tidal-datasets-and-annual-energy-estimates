"""
@Author: Eoghancurran
"""

import numpy as np
from scipy.fft import fft, ifft
import pickle


#Create function to select the top X% of frequencies by magnitude
def select_top_percent(X, top_percent=1):
    magnitudes = np.abs(X)
    n_oneside = len(X) // 2

# Calculate the threshold based on the top_percent
    threshold_index = int(n_oneside * (1 - top_percent / 100.0))
    sorted_magnitudes = np.sort(magnitudes[:n_oneside])
    threshold_value = sorted_magnitudes[threshold_index]

#mask indicating which frequencies are above the threshold
    mask = magnitudes[:n_oneside] >= threshold_value
#Stores the selected frequencies
    filtered_X = np.zeros_like(X)
    filtered_X[:n_oneside][mask] = X[:n_oneside][mask]

    return np.where(mask)[0], filtered_X[:n_oneside][mask].real, filtered_X[:n_oneside][mask].imag

#load in the arrays, will want to perform this step on the tidal, non-tidal high freq and non-tidal low freq arrays created in the decomposition step
loaded_data = np.load(r'e.g tidal freq array')
loaded_data = np.load(r'e.g non-tidal high freq array')
loaded_data = np.load(r'e.g non-tidal low freq array')

# Define parameters
N = 25920
n = np.arange(N)
sr = 1 / (60 * 5)  # Sample rate (1 sample per 5 minutes)
T = N / sr
freq = n / T
n_oneside = N // 2

# Initialize lists to hold the filtered data
filtered_indices = []
filtered_real_parts = []
filtered_imag_parts = []



# Iterate over each point and select the top x% of magnitudes
top_percent = 100  # Select top X% of magnitudes... this is adaptable
for idx in range(loaded_data.shape[0]):
    X = fft(loaded_data[idx, :])
    indices, real_parts, imag_parts = select_top_percent(X, top_percent)
    filtered_indices.append(indices)
    filtered_real_parts.append(real_parts)
    filtered_imag_parts.append(imag_parts)
    print(f"Index {idx}: Frequencies kept = {len(indices)}")

# Save the filtered data using pickle
with open(r"tidal freq file name", 'wb') as f:
with open(r"non-tidal high freq file name", 'wb') as f:
with open(r"non-tidal low freq file name", 'wb') as f:

    pickle.dump({
        'indices': filtered_indices,
        'real_parts': filtered_real_parts,
        'imag_parts': filtered_imag_parts
    }, f)

