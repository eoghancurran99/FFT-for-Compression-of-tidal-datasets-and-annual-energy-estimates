"""
@Author: Eoghancurran
"""
import numpy as np
import pandas as pd
from ocmw.core.spectral import signal_from_fft, nextpow2
from scipy.fft import fft
from ocmw.dataproc.ocmw_extract import load_ocmw_file
from pandas import date_range
from utide import reconstruct, solve
from ocmw.core.timeFuncs import datetimeFromNum

# Load data
data = load_ocmw_file(r'File Path', 'File Name')
print(data.keys())
U_data = data['depAvg_U']
V_data = data['depAvg_V']
epoch_data = data['epoch']
t_data = data['times']
loc_data = data['dataLoc']
Vpp_data = data['depAvg_Vpp']
Vtr_data = data['depAvg_Vtr']
d_data = data['depth']
surface_elevation = data['height']


# Reshape the data, with number of nodes in rows and time intervals as columns, in this case the transect had 92 points, and time interval was every 5 mins for 90 days
reshaped_data = np.zeros((92, 26010))

# Iterate over the blocks of 92 rows
num_blocks = surface_elevation.shape[0] // 92
block_size = surface_elevation.shape[1]  # 289

for i in range(num_blocks):
    start_row = i * 92
    end_row = (i + 1) * 92
    start_col = i * block_size
    end_col = (i + 1) * block_size

    # Assign the block to the corresponding position in the reshaped_data
    reshaped_data[:, start_col:end_col] = surface_elevation[start_row:end_row, :]

total_columns = reshaped_data.shape[1]
columns_to_keep = np.array([i for i in range(total_columns) if (i + 1) % 289 != 0])

reshaped_data_final = reshaped_data[:, columns_to_keep]
#save reshaped data
#np.save(r"File path", reshaped_data_final)

#obtain the mean of each point for each time-series, this will be needed later for reconstruction
mean_values = np.mean(reshaped_data_final, axis=1)

start_time = pd.to_datetime('2014-09-16 00:00')
t = date_range(start=start_time, periods=25920, freq='5min')

#Use this function to split the original time-series into tidal, high freq and low freq components
def signalDecomposition(start_time, dnum, val, lat=59.14, constit='auto', cutoff_period: float = 25.0):
    dt = start_time + pd.to_timedelta(dnum, unit='D')

    h_res = np.mean(val)
    h = val - h_res

    coef = solve(dt.values, h, lat=lat, constit=constit, method="ols", verbose=False)
    pred = reconstruct(dt.values, coef, verbose=False)

    L = len(dnum)
    t_days = dnum - np.floor(dnum[0])
    T = np.floor(np.median(np.diff(dnum) * 24.0 * 60.0 * 60.0) * 1000.0) / 1000.0
    Fs = 1.0 / T
    pw2 = nextpow2(L)
    n = np.power(2.0, pw2)
    f = Fs * np.arange(n) / n
    ts = np.arange(L) * T
    cf = 1.0 / (cutoff_period * 60.0 * 60.0)

    h_tide = pred['h']
    h_dyn = h - h_tide

    X = np.zeros([int(n), ])
    X[0:L] = h_dyn
    spec = fft(X)
    lfindx = np.where(f < cf)[0]
    if cutoff_period * 60.0 * 60.0 > T:
        h_dyn_lf = signal_from_fft(ts, f, lfindx, spec) / (n / 2)
    else:
        h_dyn_lf = signal_from_fft(ts, f, lfindx, spec) / n
    h_dyn_hf = h_dyn - h_dyn_lf

    datadict = {
        'dnum': dnum,
        't_days': t_days,
        'dtime': dt,
        'val': val,
        'val_mean': h_res,
        'tidal': pred['h'],
        'nontidal': h_dyn,
        'nontidal_lowfreq': h_dyn_lf,
        'nontidal_highfreq': h_dyn_hf,
        'model': coef
    }

    return datadict


dnum = np.arange(len(t)) / (24 * 60 / 5)

#Initialize an empty list to store the docomposition results
results = []
for i in range(reshaped_data_final.shape[0]):
    val = reshaped_data_final[i, :]
    decomposition_result = signalDecomposition(start_time, dnum, val)
    results.append(decomposition_result)

#Reconstruct tidal frequency time-series in form of an array
tidal_array = np.zeros((92,25920))
for i, result in enumerate(results):
    tidal_component = result['tidal']
    tidal_array[i, :] = tidal_component
print(tidal_array.shape)

#Reconstruct non-tidal low frequency time-series in form of an array
nontidal_lowfreq_array = np.zeros((92,25920))
for i, result in enumerate(results):
    tidal_component = result['nontidal_lowfreq']
    nontidal_lowfreq_array[i, :] = tidal_component
print(nontidal_lowfreq_array.shape)

#Reconstruct non-tidal high frequency time-series in form of an array
nontidal_highfreq_array = np.zeros((92,25920))
for i, result in enumerate(results):
    tidal_component = result['nontidal_highfreq']
    nontidal_highfreq_array[i, :] = tidal_component
print(nontidal_highfreq_array.shape)


#Save the arrays
np.save(r"File Path", tidal_array)
np.save(r"File Path", nontidal_lowfreq_array)
np.save(r"File Path", nontidal_highfreq_array)
np.save(r'File Path', mean_values)
