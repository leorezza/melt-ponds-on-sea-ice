# -*- coding: utf-8 -*-

# FIRST CELL: Import Packages, Open Data and Filtering

#%%

import h5py
import numpy as np
from numba import njit, prange
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Start measuring execution time
start_time = time.time()

# Loading and Filtering Data from HDF5 file

print("1) Loading and Filtering Data...")

def filter_photon_data(data_file, beam, surface_type_index, confidence_threshold, min_lon_hp, max_lon_hp):
    """
    Filters photon data from an ICESat-2 ATL03 dataset based on confidence level, surface type, and longitude/height constraints.
    
    Parameters:
        data_file (str): Path to the ATL03 HDF5 file.
        beam (str): Beam name (e.g., 'gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l').
        surface_type_index (int): Index for the surface type in the signal confidence array.
        confidence_threshold (int or None): Confidence threshold (0 to 4). Set to None for all photons.
        min_lon_hp (float): Minimum longitude for filtering.
        max_lon_hp (float): Maximum longitude for filtering.

    Returns:
        tuple: Filtered photon heights, latitudes, longitudes, and delta times.
    """
    # Reading data from the HDF5 file
    with h5py.File(data_file, 'r') as f:
        h_ph_raw = f[f'{beam}/heights/h_ph'][:]  # Photon heights
        lat_ph_raw = f[f'{beam}/heights/lat_ph'][:]  # Latitude positions of photons
        lon_ph_raw = f[f'{beam}/heights/lon_ph'][:]  # Longitude positions of photons
        delta_time_ph_raw = f[f'{beam}/heights/delta_time'][:]  # Time of photon return
        signal_conf_ph = f[f'{beam}/heights/signal_conf_ph'][:]  # Signal confidence level

    # Filtering 0: Confidence Level and Surface Type
    if confidence_threshold is not None:
        conf_mask = signal_conf_ph[:, surface_type_index] >= confidence_threshold
    else:
        conf_mask = np.ones(signal_conf_ph.shape[0], dtype=bool)  # Select all photons

    h_ph_filtered = h_ph_raw[conf_mask]
    lat_ph_filtered = lat_ph_raw[conf_mask]
    lon_ph_filtered = lon_ph_raw[conf_mask]
    delta_time_ph_filtered = delta_time_ph_raw[conf_mask]
    full_track_lat = lat_ph_raw[conf_mask]
    full_track_lon = lon_ph_raw[conf_mask]
    
    # Filtering 1: Longitude Range
    lon_mask = (lon_ph_filtered >= min_lon_hp) & (lon_ph_filtered <= max_lon_hp)

    # Filtering 2: Height Filter Based on Median
    median_hp = np.median(h_ph_filtered)  # Compute median of photon heights
    min_hp, max_hp = median_hp - 30, median_hp + 30  # Set height range ±30 m from median
    height_mask = (h_ph_filtered >= min_hp) & (h_ph_filtered <= max_hp)

    # Combine All Filters
    combined_mask = lon_mask & height_mask

    # Apply combined filter to all datasets
    h_ph = h_ph_filtered[combined_mask]
    lon_ph = lon_ph_filtered[combined_mask]
    lat_ph = lat_ph_filtered[combined_mask]
    delta_time_ph = delta_time_ph_filtered[combined_mask]

    return h_ph, lat_ph, lon_ph, delta_time_ph, full_track_lat, full_track_lon

# Parameters
data_file = 'ATL03_20190622061415_12980304_006_02.h5'
beam = 'gt3l'
surface_type_index = 2
confidence_threshold = 0
min_lon_hp = -56.00
max_lon_hp = -55.00

# Function call
h_ph, lat_ph, lon_ph, delta_time_ph, full_track_lat, full_track_lon  = filter_photon_data(
    data_file, beam, surface_type_index, confidence_threshold, min_lon_hp, max_lon_hp
)

print("   > Loading and Filtering Data: Completed.")
print(f"   > Filtered photons count: {len(h_ph)}")
#%%


# SECOND CELL: Density and pre-signal

#%%

# Photon density field calculation.

section_start = time.time()
print("2) Calculating photon density...")

@njit(parallel=True)
def calculate_density_batch(h_ph, lat_ph, lon_ph, sigma, anisotropy, cutoff, batch_size, overlap):
    """
    Compute photon density in batches with overlapping regions.

    Parameters:
        h_ph (numpy.ndarray): Array of photon heights.
        lat_ph (numpy.ndarray): Array of photon latitudes.
        lon_ph (numpy.ndarray): Array of photon longitudes.
        sigma (float): Standard deviation of the Gaussian kernel.
        anisotropy (float): Scaling factor for lat/lon distances.
        cutoff (float): Height difference threshold for kernel computation.
        batch_size (int): Number of photons per batch.
        overlap (int): Number of overlapping photons between adjacent batches.

    Returns:
        numpy.ndarray: Densities computed for all photons.
    """
    num_photons = len(h_ph)
    density = np.zeros(num_photons)  # Initialize the density array

    # Calculate the number of batches
    num_batches = (num_photons + batch_size - 1) // (batch_size - overlap)

    for batch_idx in prange(num_batches):
        # Define the batch range with overlap
        batch_start = batch_idx * (batch_size - overlap)
        batch_end = min(batch_start + batch_size, num_photons)
        batch_indices = np.arange(batch_start, batch_end)

        # Adjust for overlap with neighboring batches
        overlap_start = max(0, batch_start - overlap)
        overlap_end = min(num_photons, batch_end + overlap)
        extended_indices = np.arange(overlap_start, overlap_end)

        # Extract the data for the batch and the extended region
        h_ph_ext = h_ph[extended_indices]
        lat_ph_ext = lat_ph[extended_indices]
        lon_ph_ext = lon_ph[extended_indices]

        # Compute density for photons in the batch
        for i in batch_indices:
            distances = np.sqrt(
                (lat_ph_ext - lat_ph[i]) ** 2 + (lon_ph_ext - lon_ph[i]) ** 2
            ) * anisotropy
            height_diffs = np.abs(h_ph_ext - h_ph[i])

            # Gaussian kernel
            kernel = np.exp(-(distances ** 2 + height_diffs ** 2) / (2 * sigma ** 2))

            # Exclude contributions beyond the cutoff threshold
            kernel[height_diffs > (cutoff * sigma)] = 0

            # Compute the density for the current photon
            density[i] = np.sum(kernel)

    return density

# Assuming h_ph, lat_ph, lon_ph, delta_time_ph have been preloaded
sigma, anisotropy, cutoff = 3, 20, 2
batch_size = 1000  # Number of photons per batch
overlap = 250  # Overlap between batches

def calculate_density_with_progress(h_ph, lat_ph, lon_ph, sigma, anisotropy, cutoff, batch_size, overlap):
    num_photons = len(h_ph)
    density = np.zeros(num_photons)  # Initialize the density array

    # Calculate the number of batches
    num_batches = (num_photons + batch_size - 1) // (batch_size - overlap)

    # Display progress bar using tqdm
    with tqdm(total=num_batches, desc="Processing Batches", unit="batch") as pbar:
        for batch_idx in range(num_batches):
            # Define the batch range with overlap
            batch_start = batch_idx * (batch_size - overlap)
            batch_end = min(batch_start + batch_size, num_photons)
            batch_indices = np.arange(batch_start, batch_end)

            # Adjust for overlap with neighboring batches
            overlap_start = max(0, batch_start - overlap)
            overlap_end = min(num_photons, batch_end + overlap)
            extended_indices = np.arange(overlap_start, overlap_end)

            # Extract the data for the batch and the extended region
            h_ph_ext = h_ph[extended_indices]
            lat_ph_ext = lat_ph[extended_indices]
            lon_ph_ext = lon_ph[extended_indices]

            # Compute density for photons in the batch
            for i in batch_indices:
                distances = np.sqrt(
                    (lat_ph_ext - lat_ph[i]) ** 2 + (lon_ph_ext - lon_ph[i]) ** 2
                ) * anisotropy
                height_diffs = np.abs(h_ph_ext - h_ph[i])

                # Gaussian kernel
                kernel = np.exp(-(distances ** 2 + height_diffs ** 2) / (2 * sigma ** 2))

                # Exclude contributions beyond the cutoff threshold
                kernel[height_diffs > (cutoff * sigma)] = 0

                # Compute the density for the current photon
                density[i] = np.sum(kernel)

            # Update progress bar
            pbar.update(1)

    return density

density = calculate_density_with_progress(h_ph, lat_ph, lon_ph, sigma, anisotropy, cutoff, batch_size, overlap)

print(f"   > Density calculation: Completed. Time taken: {time.time() - section_start:.2f} seconds.")


# Signal and Noise Slab Separation

print("3) Separating Signal and Noise...")

def separate_signal_and_noise(h_ph, lat_ph, lon_ph, density, bin_size_along, bin_size_height, slab_thickness, tbin, threshold_offset, q_quantile):
    """
    Optimized function to separate signal photons from noise using a histogram-based method
    with Numba acceleration and a progress bar.

    Parameters:
        h_ph (numpy.ndarray): Photon height array.
        lat_ph (numpy.ndarray): Photon latitude array.
        lon_ph (numpy.ndarray): Photon longitude array.
        density (numpy.ndarray): Density values corresponding to each photon.
        bin_size_along (float): Along-track bin size in degrees.
        bin_size_height (float): Height bin size in meters.
        slab_thickness (float): Thickness of signal and noise slabs in meters.
        tbin (float): Along-track bin size for signal/noise separation in degrees.
        threshold_offset (float): Offset added to the noise threshold.
        q_quantile (float): Quantile value for second signal threshold.

    Returns:
        tuple: Filtered signal photon positions (latitude, longitude, height, density).
    """
    
   
    # Step 1: Histogram Binning
    _, edges_along = np.histogram(lon_ph, bins=np.arange(min(lon_ph), max(lon_ph) + bin_size_along, bin_size_along))
    hist_height, edges_height = np.histogram(h_ph, bins=np.arange(min(h_ph), max(h_ph) + bin_size_height, bin_size_height))

    # Step 2: Signal and Noise Slab Identification
    signal_slab_center_idx = np.argmax(hist_height)
    signal_slab_center = edges_height[signal_slab_center_idx]
    signal_slab = [signal_slab_center - slab_thickness / 2, signal_slab_center + slab_thickness / 2]
    noise_slab = [signal_slab[1], signal_slab[1] + slab_thickness]

    # Step 3: Signal and Noise Separation
    
    # Compute along-track distance bins
    bins = np.arange(lon_ph.min(), lon_ph.max(), tbin)
    bin_indices = np.digitize(lon_ph, bins)

    # Initialize final signal photon set
    final_signal_photons = np.zeros_like(h_ph, dtype=bool)

    # Iterate through each along-track bin with a progress bar
    for b in tqdm(range(1, len(bins)), desc="Processing bins"):
        in_bin = bin_indices == b
        if not np.any(in_bin):
            continue  # Skip empty bins

        # Noise slab density calculation
        noise_photons = (h_ph[in_bin] >= noise_slab[0]) & (h_ph[in_bin] <= noise_slab[1])
        fmax_noise_s = density[in_bin][noise_photons].max() if np.any(noise_photons) else 0

        # First threshold
        t_s1 = fmax_noise_s + threshold_offset

        # Signal candidates in the signal slab
        signal_candidates = (density[in_bin] > t_s1) & (h_ph[in_bin] >= signal_slab[0]) & (h_ph[in_bin] <= signal_slab[1])
        Ts = density[in_bin][signal_candidates]

        if len(Ts) > 0:
            # Second threshold
            t_s2 = np.quantile(Ts, q_quantile)
            final_signal_photons[in_bin] = density[in_bin] > t_s2

    # Extract final signal photon positions and density 
    photon_lat_signal = lat_ph[final_signal_photons]
    photon_lon_signal = lon_ph[final_signal_photons]
    photon_height_signal = h_ph[final_signal_photons]
    density_signal = density[final_signal_photons]


    return photon_lat_signal, photon_lon_signal, photon_height_signal, density_signal

# Call the optimized function
photon_lat_signal, photon_lon_signal, photon_height_signal, density_signal = separate_signal_and_noise(
    h_ph, lat_ph, lon_ph, density,
    bin_size_along=0.003552,  # 50 m in longitude degrees (°)
    bin_size_height=10,       # 10 m
    slab_thickness=30,        # 30 m
    tbin=0.000355,            # 5 m in longitude degrees (°)
    threshold_offset=1,
    q_quantile=0.15
)

# Plot Pre-signal
plt.figure()
plt.scatter(lon_ph, h_ph, s=1.5, color='k', label='Noise')
plt.scatter(photon_lon_signal, photon_height_signal, s=2.5, marker='o', color='green', label='Signal')
plt.title('Pre-signal after noise/signal separation')
plt.xlabel('Longitude (degrees °)')
plt.ylabel('Photon Heights (m)')
plt.legend(loc='upper right')
plt.grid(True)

print("   > Separation of Signal and Noise. Filtered Pre-Signal: Completed.")


#%%


# THIRD CELL: melt pond detection

#%%

# Melt-pond detection: compute vertical histograms for melt ponds


def compute_vertical_histograms(photon_lon_signal, photon_height_signal, mp_binh, mp_binv):
    """
    Computes and filters vertical histograms for melt-pond detection.

    Parameters:
        photon_lon_signal (numpy.ndarray): Longitudes of signal photons.
        photon_height_signal (numpy.ndarray): Heights of signal photons.
        mp_binh (float): Horizontal bin size in degrees (e.g., 0.0003 ~ 25 m).
        mp_binv (float): Vertical bin size in meters (e.g., 0.2 m).

    Returns:
        numpy.ndarray: Filtered vertical histogram array.
    """
    print("4) Calculating vertical histogram...")

    # Step 1: Compute horizontal and vertical ranges
    lon_min, lon_max = np.min(photon_lon_signal), np.max(photon_lon_signal)
    height_min, height_max = np.min(photon_height_signal), np.max(photon_height_signal)

    # Step 2: Create bin edges
    bin_edges_horizontal = np.arange(lon_min, lon_max + mp_binh, mp_binh)
    bin_edges_vertical = np.arange(height_min, height_max + mp_binv, mp_binv)

    # Step 3: Initialize histogram
    H = np.zeros((len(bin_edges_horizontal) - 1, len(bin_edges_vertical) - 1))

    # Step 4: Populate histogram by binning signal photons
    for i in range(len(bin_edges_horizontal) - 1):
        # Find photons in the current horizontal bin
        in_horizontal_bin = (photon_lon_signal >= bin_edges_horizontal[i]) & (photon_lon_signal < bin_edges_horizontal[i + 1])
        vertical_photons = photon_height_signal[in_horizontal_bin]

        if len(vertical_photons) > 0:
            H[i, :], _ = np.histogram(vertical_photons, bins=bin_edges_vertical)

    print("5) Applying binomial filter...")

    # Step 5: Apply binomial filter along the vertical axis (height bins)
    H_filtered = np.copy(H)

    for i in range(H.shape[0]):  # Iterate through horizontal bins
        for x in range(H.shape[1]):  # Iterate through vertical bins
            # Binomial filtering weights
            H_filtered[i, x] = (
                (0.0625 * H[i, x - 2] if x - 2 >= 0 else 0) +
                (0.25 * H[i, x - 1] if x - 1 >= 0 else 0) +
                (0.375 * H[i, x]) +
                (0.25 * H[i, x + 1] if x + 1 < H.shape[1] else 0) +
                (0.0625 * H[i, x + 2] if x + 2 < H.shape[1] else 0)
            )

    print("   > Application of Binomial Filter to vertical histograms: Completed.")
    print("   > Calculation of vertical histograms: Completed.")

    return H_filtered, bin_edges_horizontal, bin_edges_vertical

# Call the Function "compute_vertical_histograms" with the parameters from the paper
mp_binh = 0.0003  # Horizontal bin size in longitude degrees ° (approx. 25 m)
mp_binv = 0.2     # Vertical bin size (0.2 m)

# Compute and filter vertical histograms
H_new, bin_edges_horizontal, bin_edges_vertical = compute_vertical_histograms(photon_lon_signal, photon_height_signal, mp_binh, mp_binv)

# Melt-Pond Detection: finding peaks and bifurcation of top/bottom signal 

print("6) Finding Peaks and bottom/top surface slabs in the vertical histograms...")
print("7) Extracting Pre-Signal Photons for bottom/top slabs...")

def detect_peaks_and_define_slabs(H_new, bin_edges_horizontal, bin_edges_vertical, lon_start, lon_end, min_photon_count, min_prominence, min_pond_depth):
    """
    Detects peaks, defines slabs, and computes vertical slab boundaries for selected horizontal bins.

    Parameters:
        H_new (numpy.ndarray): 2D histogram array after binomial filtering.
        bin_edges_horizontal (numpy.ndarray): Edges of horizontal bins (longitude).
        bin_edges_vertical (numpy.ndarray): Edges of vertical bins (photon height).
        lon_start (float): Start of the longitude range for detection.
        lon_end (float): End of the longitude range for detection.
        min_photon_count (int): Minimum photon count for peak detection.
        min_prominence (float): Minimum prominence for peak detection.
        min_pond_depth (float): Minimum separation between peaks.

    Returns:
        dict: Dictionary with horizontal bin centers as keys and slab boundaries
        as values expressed in photon heights (m). Each value includes peaks, 
        saddle point, and slab information distinguishing between cases with 1 
        or 2 peaks in each horizontal bin:
            Example: {bin_center: {"peaks": [peak1, peak2],
                                    "saddle_point": saddle_height,
                                    "upper_slab": (start_height, end_height),
                                    "lower_slab": (start_height, end_height),
                                    "top_slab": (start_height, end_height)}}
    """
    selected_bins = (bin_edges_horizontal[:-1] >= lon_start) & (bin_edges_horizontal[1:] <= lon_end)
    selected_H = H_new[selected_bins, :]
    selected_bin_edges_horizontal = bin_edges_horizontal[:-1][selected_bins]

    slab_boundaries = {}  # Dictionary to store slab boundaries for each horizontal bin

    for i, hist_data in enumerate(selected_H):
        bin_center = selected_bin_edges_horizontal[i]

        # Detect peaks in the histogram
        peaks, _ = find_peaks(hist_data, height=min_photon_count, prominence=min_prominence, distance=min_pond_depth)
        slabs = {"peaks": [], "saddle_point": None, "upper_slab": None, "lower_slab": None, "top_slab": None}

        if len(peaks) >= 2:
            # Sort peaks by photon height (descending order)
            sorted_peaks = sorted(peaks, key=lambda p: bin_edges_vertical[p], reverse=True)
            peak1, peak2 = sorted_peaks[:2]

            # Saddle point (minimum bin) between peaks
            saddle_bin = np.argmin(hist_data[min(peak1, peak2):max(peak1, peak2)]) + min(peak1, peak2)
            saddle_height = bin_edges_vertical[saddle_bin]

            # Define slabs
            upper_slab = (saddle_bin, peak1 + (peak1 - saddle_bin))  # Top Surface
            lower_slab = (saddle_bin, peak2)                         # Bottom Surface

            # Adjust bottom slab lower boundary
            while lower_slab[1] > 0 and hist_data[lower_slab[1] - 1] > 0:
                lower_slab = (lower_slab[0], lower_slab[1] - 1)

            # Convert slab bin indices to photon heights
            slabs["peaks"] = [bin_edges_vertical[peak2], bin_edges_vertical[peak1]]
            slabs["saddle_point"] = saddle_height
            slabs["upper_slab"] = (bin_edges_vertical[upper_slab[0]], bin_edges_vertical[upper_slab[1]])
            slabs["lower_slab"] = (bin_edges_vertical[lower_slab[1]], bin_edges_vertical[lower_slab[0]])

        elif len(peaks) == 1:
            # Single Peak Found
            peak = peaks[0]

            # Include all bins in the vertical range of non-zero photon counts
            min_vertical_bin = np.min(np.where(hist_data > 0))  # First non-zero vertical bin
            max_vertical_bin = np.max(np.where(hist_data > 0))  # Last non-zero vertical bin

            # Define the top slab using the actual range of bins for the one peak case
            top_slab = (min_vertical_bin, max_vertical_bin)

            # Convert slab bin indices to photon heights
            slabs["peaks"] = [bin_edges_vertical[peak]]
            slabs["top_slab"] = (bin_edges_vertical[top_slab[0]], bin_edges_vertical[top_slab[1]])

        slab_boundaries[bin_center] = slabs

    return slab_boundaries

# Call Function and Detect Peaks and Define Slabs
slab_boundaries = detect_peaks_and_define_slabs(
    H_new, bin_edges_horizontal, bin_edges_vertical,
    lon_start= min_lon_hp, lon_end= max_lon_hp,
    min_photon_count=3,
    min_prominence=2,
    min_pond_depth=2
)

print("   > Finding Peaks and bottom/top surface slabs: Completed.")
print("   > Pre-Signal Photons for bottom/top slabs Extraction: Completed.")

# Melt-Pond Detection: Defining Slab Ranges Along-Track

def define_slab_ranges(slab_boundaries, bin_edges_horizontal, lon_start, lon_end):
    """
    Defines slab ranges (upper, lower, top) for each longitude bin in the specified range.

    Parameters:
        slab_boundaries (dict): Dictionary containing slab boundaries for each horizontal bin.
                                Each entry includes peaks, saddle point, and slab information.
        bin_edges_horizontal (numpy.ndarray): Edges of horizontal bins (longitude).
        lon_start (float): Start of the longitude range.
        lon_end (float): End of the longitude range.

    Returns:
        dict: Dictionary with longitude ranges as keys and slab ranges as values.
              Example: { "lon_range": {"upper_slab": (start_height, end_height),
                                        "lower_slab": (start_height, end_height),
                                        "top_slab": (start_height, end_height)} }
    """
    print("8) Defining slab ranges along the track...")

    slab_ranges = {}  # Dictionary to store slab ranges for each longitude interval

    for i, (bin_center, slabs) in enumerate(slab_boundaries.items()):
        # Identify the edges for the current bin
        if i < len(bin_edges_horizontal) - 1:
            bin_left = bin_edges_horizontal[i]
            bin_right = bin_edges_horizontal[i + 1]

            # Skip bins outside the specified range
            if bin_right < lon_start or bin_left > lon_end:
                continue

            # Store slab ranges for this longitude bin
            slab_ranges[f"{bin_left:.5f} to {bin_right:.5f}"] = {
                "upper_slab": slabs["upper_slab"],
                "lower_slab": slabs["lower_slab"],
                "top_slab": slabs["top_slab"]
            }

    print("   > Defining Slab ranges along-track: Completed.")
    return slab_ranges

# Define slab ranges for a specific longitude range
slab_ranges = define_slab_ranges(slab_boundaries, bin_edges_horizontal, lon_start=min_lon_hp, lon_end=max_lon_hp)

# Melt-pond detection: defining slab ranges along-track

def correct_slab_ranges(slab_ranges):
    """
    Corrects the slab ranges based on a moving window of 7 bins.

    Conditions:
    - For each horizontal bin, check a moving window of 7 bins (centered on the current bin).
    - If there are at least two consecutive bins (including the selected bin) within the window
      that have valid `upper_slab` and `lower_slab`, the correction is not applied.
    - Otherwise, the `upper_slab` and `lower_slab` are replaced with a `top_slab`:
      [lower_slab[0], upper_slab[1]].

    Parameters:
        slab_ranges (dict): Dictionary with longitude ranges as keys and slab ranges as values.
                            Example: {
                                "lon_range": {
                                    "upper_slab": (start_height, end_height),
                                    "lower_slab": (start_height, end_height),
                                    "top_slab": (start_height, end_height)
                                }
                            }

    Returns:
        dict: Corrected slab ranges with the updated slab definitions.
    """
    print("9) Correcting slab ranges based on a moving window of 7 bins...")

    corrected_slab_ranges = {}  # Dictionary to store corrected slab ranges
    slab_keys = list(slab_ranges.keys())  # List of longitude keys for consecutive checks
    num_bins = len(slab_keys)

    # Iterate through slab ranges and apply correction
    for i, key in enumerate(slab_keys):
        current_slab = slab_ranges[key]
        upper_slab = current_slab.get("upper_slab")
        lower_slab = current_slab.get("lower_slab")

        # Check if both upper_slab and lower_slab exist and are not None
        if upper_slab and lower_slab:
            # Define the moving window: 7 bins centered on the current bin
            window_start = max(0, i - 3)  # 3 bins before
            window_end = min(num_bins, i + 4)  # 3 bins after

            # Check for at least three consecutive valid bins within the window
            valid_bins = [
                slab_ranges[slab_keys[j]].get("upper_slab") and slab_ranges[slab_keys[j]].get("lower_slab")
                for j in range(window_start, window_end)
            ]

            # Check if there are two consecutive True values in valid_bins
            consecutive_count = 0
            has_three_consecutive = False
            for is_valid in valid_bins:
                if is_valid:
                    consecutive_count += 1
                    if consecutive_count >= 3:
                        has_three_consecutive = True
                        break
                else:
                    consecutive_count = 0

            if has_three_consecutive:
                # Keep the bifurcation
                corrected_slab_ranges[key] = current_slab
            else:
                # Correct to a single top slab range
                corrected_slab_ranges[key] = {
                    "top_slab": (lower_slab[0], upper_slab[1]),
                    "upper_slab": None,
                    "lower_slab": None,
                }
        else:
            # No bifurcation: Copy the slab ranges directly
            corrected_slab_ranges[key] = {
                "top_slab": current_slab.get("top_slab"),
                "upper_slab": None,
                "lower_slab": None,
            }

    print("   > Slab range correction completed.")
    return corrected_slab_ranges

# Correct the slab ranges
slab_ranges = correct_slab_ranges(slab_ranges)


# Plot vertical histograms

def plot_vertical_histograms_with_annotations(H_new, bin_edges_horizontal, bin_edges_vertical, lon_start, lon_end, min_photon_count, min_prominence, min_pond_depth):
    """
    Plots vertical histograms for selected longitude bins and annotates them with peaks, saddle points, and slabs.

    Parameters:
        H_new (numpy.ndarray): 2D histogram array after binomial filtering.
        bin_edges_horizontal (numpy.ndarray): Edges of horizontal bins (longitude).
        bin_edges_vertical (numpy.ndarray): Edges of vertical bins (photon height).
        lon_start (float): Start of the longitude range for plotting.
        lon_end (float): End of the longitude range for plotting.
        min_photon_count (int): Minimum photon count for peak detection.
        min_prominence (float): Minimum prominence for peak detection.
        min_pond_depth (float): Minimum separation between peaks.

    Returns:
        None: Plots vertical histograms with annotations.
    """
    print("10) Plotting Vertical Histograms with Peaks, Saddle Point, and Slabs for a given range...")

    # Select bins corresponding to the longitude range
    selected_bins = (bin_edges_horizontal[:-1] >= lon_start) & (bin_edges_horizontal[1:] <= lon_end)
    selected_H = H_new[selected_bins, :]
    selected_bin_edges_horizontal = bin_edges_horizontal[:-1][selected_bins]

    # Function to annotate histogram
    def annotate_histogram(hist_data, peaks, bin_edges_vertical):
        if len(peaks) >= 2:
            # Sort peaks by photon height (descending order)
            sorted_peaks = sorted(peaks, key=lambda p: bin_edges_vertical[p], reverse=True)
            peak1, peak2 = sorted_peaks[:2]

            # Saddle point (minimum bin) between peaks
            saddle_bin = np.argmin(hist_data[min(peak1, peak2):max(peak1, peak2)]) + min(peak1, peak2)
            d = peak1 - saddle_bin  # Define distance d

            # Define slabs
            upper_slab = (saddle_bin, peak1 + d)  # Top Surface
            lower_slab = (saddle_bin, peak2)      # Bottom Surface

            # Adjust bottom slab lower boundary
            while lower_slab[1] > 0 and hist_data[lower_slab[1] - 1] > 0:
                lower_slab = (lower_slab[0], lower_slab[1] - 1)

            # Annotate peaks
            plt.axvline(x=bin_edges_vertical[peak1], color='b', linestyle='--', label='Peak 1 (Top Surface)')
            plt.axvline(x=bin_edges_vertical[peak2], color='g', linestyle='--', label='Peak 2 (Bottom Surface)')
            plt.text(bin_edges_vertical[peak1], hist_data[peak1] + 2, f'Peak 1: {bin_edges_vertical[peak1]:.2f} m', 
                      color='b', ha='center', fontsize=9)
            plt.text(bin_edges_vertical[peak2], hist_data[peak2] + 2, f'Peak 2: {bin_edges_vertical[peak2]:.2f} m', 
                      color='g', ha='center', fontsize=9)

            # Annotate saddle point
            plt.axvline(x=bin_edges_vertical[saddle_bin], color='purple', linestyle=':', label='Saddle Point')
            plt.text(bin_edges_vertical[saddle_bin], hist_data[saddle_bin] + 2, 
                      f'Saddle: {bin_edges_vertical[saddle_bin]:.2f} m', color='purple', ha='center', fontsize=9)

            # Highlight slabs
            plt.axvspan(bin_edges_vertical[upper_slab[0]], bin_edges_vertical[upper_slab[1]], 
                        color='blue', alpha=0.2, label='Upper Slab')
            plt.axvspan(bin_edges_vertical[lower_slab[0]], bin_edges_vertical[lower_slab[1]], 
                        color='green', alpha=0.2, label='Lower Slab')

        elif len(peaks) == 1:
            # Single Peak Found
            peak = peaks[0]

            # Include all bins in the horizontal bin
            min_vertical_bin = np.min(np.where(hist_data > 0))  # First non-zero vertical bin
            max_vertical_bin = np.max(np.where(hist_data > 0))  # Last non-zero vertical bin

            # Define the top slab using the actual range of bins for the one peak case
            top_slab = (min_vertical_bin, max_vertical_bin)

            plt.axvline(x=bin_edges_vertical[peak], color='r', linestyle='--', label='Single Peak')
            plt.text(bin_edges_vertical[peak], hist_data[peak] + 2, f'Peak: {bin_edges_vertical[peak]:.2f} m', 
                      color='r', ha='center', fontsize=9)
            plt.axvspan(bin_edges_vertical[top_slab[0]], bin_edges_vertical[top_slab[1]], 
                        color='blue', alpha=0.2, label='Top Slab')

    # Plot the vertical histograms for each selected bin
    for i, hist_data in enumerate(selected_H):
        peaks, properties = find_peaks(hist_data, height=min_photon_count, prominence=min_prominence, distance=min_pond_depth)

        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges_vertical[:-1], hist_data, width=np.diff(bin_edges_vertical)[0], 
                edgecolor='k', align='edge', label='Photon Counts')

        # Annotate histogram
        annotate_histogram(hist_data, peaks, bin_edges_vertical)

        # Add titles and labels
        plt.title(f'Vertical Histogram for Longitude Bin {selected_bin_edges_horizontal[i]:.5f}°')
        plt.xlabel('Photon Height (m)')
        plt.ylabel('Counts')
        plt.ylim(0, max(hist_data) + 10)  # Adjust y-limit to fit peaks
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.show()

print("   > Plotting of vertical histograms with peaks, saddle point, and slabs: Completed.")
  
# plot_vertical_histograms_with_annotations(
#     H_new, bin_edges_horizontal, bin_edges_vertical,
#     lon_start=-55.686, lon_end=-55.682,  # Example longitude range (first melt pond)
#     min_photon_count=3,
#     min_prominence=2,
#     min_pond_depth=2
# )


def plot_slab_boundaries_along_track(slab_boundaries, bin_edges_horizontal, lon_start, lon_end):
    """
    Plots slab boundaries as color-shaded areas along the longitude axis without white spaces between bins.

    Parameters:
        slab_boundaries (dict): Dictionary containing slab boundaries for each horizontal bin.
                                Each entry includes peaks, saddle point, and slab information.
        bin_edges_horizontal (numpy.ndarray): Edges of horizontal bins (longitude).
        lon_start (float): Start of the longitude range for plotting.
        lon_end (float): End of the longitude range for plotting.

    Returns:
        None: Displays the plot.
    """
    print("11) Plotting slab boundaries along the track...")

    plt.figure(figsize=(12, 6))

    for i, (bin_center, slabs) in enumerate(slab_boundaries.items()):
        # Identify the edges for the current bin
        if i < len(bin_edges_horizontal) - 1:
            bin_left = bin_edges_horizontal[i]
            bin_right = bin_edges_horizontal[i + 1]

            if bin_right < lon_start or bin_left > lon_end:
                continue  # Skip bins outside the specified range

            # Plot upper slab
            if slabs["upper_slab"]:
                upper_start, upper_end = slabs["upper_slab"]
                plt.fill_betweenx([upper_start, upper_end], bin_left, bin_right,
                                  color='blue', alpha=0.3, label='Upper Slab' if 'Upper Slab' not in plt.gca().get_legend_handles_labels()[1] else "")

            # Plot lower slab
            if slabs["lower_slab"]:
                lower_start, lower_end = slabs["lower_slab"]
                plt.fill_betweenx([lower_start, lower_end], bin_left, bin_right,
                                  color='green', alpha=0.3, label='Lower Slab' if 'Lower Slab' not in plt.gca().get_legend_handles_labels()[1] else "")

            # Plot top slab for single peak cases
            if slabs["top_slab"]:
                top_start, top_end = slabs["top_slab"]
                plt.fill_betweenx([top_start, top_end], bin_left, bin_right,
                                  color='red', alpha=0.3, label='Top Slab' if 'Top Slab' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Customize the plot
    plt.title("Slab Boundaries Along Longitude")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Photon Height (m)")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlim(lon_start, lon_end)
    plt.show()

    print("   > Plotting Slabs: Completed.")

# plot_slab_boundaries_along_track(slab_boundaries, bin_edges_horizontal, lon_start= min_lon_hp, lon_end= max_lon_hp)


# Melt-Pond Detection: Classify Photons as Top (One-Peak Top-Surface Signal), Lower or Upper (for Melt Pond)

def classify_signal_photons(photon_lon_signal, photon_height_signal, density_signal, slab_ranges):
    """
    Classifies signal photons and their densities into 'top', 'upper', or 'lower' groups based on slab ranges.

    Parameters:
        photon_lon_signal (numpy.ndarray): Longitudes of signal photons.
        photon_height_signal (numpy.ndarray): Heights of signal photons.
        density_signal (numpy.ndarray): Densities of signal photons.
        slab_ranges (dict): Dictionary of slab ranges keyed by longitude intervals.
                            Example: {"lon_left to lon_right": {"upper_slab": (start_height, end_height),
                                                               "lower_slab": (start_height, end_height),
                                                               "top_slab": (start_height, end_height)}}

    Returns:
        dict: Dictionary containing classified photons and densities:
              Example: {
                  "top": {"lon": [...], "height": [...], "density": [...]},
                  "upper": {"lon": [...], "height": [...], "density": [...]},
                  "lower": {"lon": [...], "height": [...], "density": [...]}
              }
    """
    print("12) Classifying signal photons and densities into groups...")

    # Initialize groups
    classified_photons = {
        "top": {"lon": [], "height": [], "density": []},
        "upper": {"lon": [], "height": [], "density": []},
        "lower": {"lon": [], "height": [], "density": []}
    }

    # Iterate over slab ranges
    for lon_range, slabs in slab_ranges.items():
        # Parse longitude range
        lon_left, lon_right = map(float, lon_range.split(" to "))

        # Identify photons within this longitude range
        in_lon_range = (photon_lon_signal >= lon_left) & (photon_lon_signal <= lon_right)
        photons_in_range = photon_height_signal[in_lon_range]
        photon_lons_in_range = photon_lon_signal[in_lon_range]
        densities_in_range = density_signal[in_lon_range]

        # Classify photons and densities based on their height with respect to the slabs
        for lon, height, density in zip(photon_lons_in_range, photons_in_range, densities_in_range):
            if slabs["top_slab"] and slabs["top_slab"][0] <= height <= slabs["top_slab"][1]:
                classified_photons["top"]["lon"].append(lon)
                classified_photons["top"]["height"].append(height)
                classified_photons["top"]["density"].append(density)
            elif slabs["upper_slab"] and slabs["upper_slab"][0] <= height <= slabs["upper_slab"][1]:
                classified_photons["upper"]["lon"].append(lon)
                classified_photons["upper"]["height"].append(height)
                classified_photons["upper"]["density"].append(density)
            elif slabs["lower_slab"] and slabs["lower_slab"][0] <= height <= slabs["lower_slab"][1]:
                classified_photons["lower"]["lon"].append(lon)
                classified_photons["lower"]["height"].append(height)
                classified_photons["lower"]["density"].append(density)

    # Convert lists to numpy arrays for easier processing later
    for group in classified_photons:
        classified_photons[group]["lon"] = np.array(classified_photons[group]["lon"])
        classified_photons[group]["height"] = np.array(classified_photons[group]["height"])
        classified_photons[group]["density"] = np.array(classified_photons[group]["density"])

    print("   > Photon and density classification: Completed.")
    return classified_photons


# Classify signal photons
classified_photons = classify_signal_photons(photon_lon_signal, photon_height_signal, density_signal, slab_ranges)


# Plot classified photons


# def plot_classified_photons_with_density(classified_photons):
#     """
#     Plots a scatter plot of photon height vs. longitude with classified photons in different colors.
#     Additionally, creates density plots for top and lower photons, and top and upper photons.

#     Parameters:
#         classified_photons (dict): Dictionary containing classified photons and densities:
#               Example: {
#                   "top": {"lon": [...], "height": [...], "density": [...]},
#                   "upper": {"lon": [...], "height": [...], "density": [...]},
#                   "lower": {"lon": [...], "height": [...], "density": [...]}
#               }

#     Returns:
#         None: Displays scatter and density plots.
#     """
#     print("10) Plotting classified photons...")

#     # Scatter plot
#     plt.figure(figsize=(12, 6))

#     # Plot each group with a different color
#     if len(classified_photons["top"]["lon"]) > 0:
#         plt.scatter(classified_photons["top"]["lon"], classified_photons["top"]["height"],
#                     c='green', label='Top Photons', s=5, alpha=0.5)

#     if len(classified_photons["upper"]["lon"]) > 0:
#         plt.scatter(classified_photons["upper"]["lon"], classified_photons["upper"]["height"],
#                     c='blue', label='Upper Photons', s=5, alpha=0.5)

#     if len(classified_photons["lower"]["lon"]) > 0:
#         plt.scatter(classified_photons["lower"]["lon"], classified_photons["lower"]["height"],
#                     c='red', label='Lower Photons', s=5, alpha=0.5)

#     # Customize the scatter plot
#     plt.title("Classified Photons: Height vs. Longitude")
#     plt.xlabel("Longitude (°)")
#     plt.ylabel("Photon Height (m)")
#     plt.grid(True)
#     plt.legend(loc='upper right')
#     plt.show()

#     print("   > Scatter plot of classified photons: Completed.")

#     # Density plot
    
#     if len(classified_photons["top"]["density"]) > 0 and len(classified_photons["lower"]["density"]) > 0:
#         plt.figure(figsize=(12, 6))

#         # Combine top and lower photons
#         lon_combined = np.concatenate([classified_photons["top"]["lon"], classified_photons["lower"]["lon"]])
#         height_combined = np.concatenate([classified_photons["top"]["height"], classified_photons["lower"]["height"]])
#         density_combined = np.concatenate([classified_photons["top"]["density"], classified_photons["lower"]["density"]])

#         # Scatter Plot
#         plt.scatter(lon_combined, height_combined, c=density_combined, cmap='coolwarm', s=10, alpha=0.7)

#         # Customize the density plot
#         plt.colorbar(label='Normalized Density')
#         plt.title("Density Plot: Top and Lower Photons")
#         plt.xlabel("Longitude (°)")
#         plt.ylabel("Photon Height (m)")
#         plt.grid(True)
#         plt.show()

#         print("   > Density plot of top and lower photons: Completed.")

#     # Density plot: top and upper photons
    
#     if len(classified_photons["top"]["density"]) > 0 and len(classified_photons["upper"]["density"]) > 0:
#         plt.figure(figsize=(12, 6))

#         # Combine top and upper photons
#         lon_combined = np.concatenate([classified_photons["top"]["lon"], classified_photons["upper"]["lon"]])
#         height_combined = np.concatenate([classified_photons["top"]["height"], classified_photons["upper"]["height"]])
#         density_combined = np.concatenate([classified_photons["top"]["density"], classified_photons["upper"]["density"]])

#         # Scatter Plot
#         plt.scatter(lon_combined, height_combined, c=density_combined, cmap='coolwarm', s=10, alpha=0.7)

#         # Customize the density plot
#         plt.colorbar(label='Normalized Density')
#         plt.title("Density Plot: Top and Upper Photons")
#         plt.xlabel("Longitude (°)")
#         plt.ylabel("Photon Height (m)")
#         plt.grid(True)
#         plt.show()

#         print("   > Density plot of top and upper photons: Completed.")

# # Plot classified photons
# plot_classified_photons_with_density(classified_photons)

#%%

# FOURTH CELL: interpolation and pond statistics)

#%%

# def interpolate_combined_photon_profiles(classified_photons, bin_size, adaptive_resolution_factor):
#     """
#     Interpolates the photon profiles for two combined datasets:
#     - Top and Lower photons.
#     - Top and Upper photons.

#     Parameters:
#         classified_photons (dict): Dictionary containing classified photons:
#                                    Example: {
#                                        "top": {"lon": [...], "height": [...], "density": [...]},
#                                        "upper": {"lon": [...], "height": [...], "density": [...]},
#                                        "lower": {"lon": [...], "height": [...], "density": [...]}
#                                    }
#         bin_size (float): Bin size for grouping photons along the longitude axis.
#         adaptive_resolution_factor (float): Factor to adjust the resolution of interpolation dynamically.

#     Returns:
#         dict: Dictionary containing the interpolated curves:
#               Example: {
#                   "top_lower": {"lon": [...], "height": [...]},
#                   "top_upper": {"lon": [...], "height": [...]}
#               }
#     """
#     print("13) Interpolating combined photon profiles using median density weighting...")

#     interpolated_profiles = {"top_lower": {"lon": [], "height": []},
#                              "top_upper": {"lon": [], "height": []}}

#     for pair_name, groups in [("top_lower", ["top", "lower"]), ("top_upper", ["top", "upper"])]:
#         # Combine data from the specified groups
#         combined_lon = np.concatenate([classified_photons[groups[0]]["lon"], classified_photons[groups[1]]["lon"]])
#         combined_height = np.concatenate([classified_photons[groups[0]]["height"], classified_photons[groups[1]]["height"]])
#         combined_density = np.concatenate([classified_photons[groups[0]]["density"], classified_photons[groups[1]]["density"]])

#         if len(combined_lon) == 0:
#             print(f"   > No data points available for {pair_name} interpolation.")
#             continue

#         # Bin the data along longitude
#         bins = np.arange(np.min(combined_lon), np.max(combined_lon) + bin_size, bin_size)
#         bin_indices = np.digitize(combined_lon, bins)

#         # Compute median-density-weighted average heights per bin
#         bin_centers = (bins[:-1] + bins[1:]) / 2
#         binned_heights = np.zeros(len(bin_centers))

#         for i in range(len(bin_centers)):
#             in_bin = bin_indices == (i + 1)
#             if np.any(in_bin):
#                 # Compute median density for the current bin
#                 median_density = np.median(combined_density[in_bin])

#                 # Apply median density as the weight for height averaging
#                 heights_in_bin = combined_height[in_bin]
#                 densities_in_bin = combined_density[in_bin]

#                 # Compute weighted average using the median density
#                 weights = np.full(len(heights_in_bin), median_density)
#                 binned_heights[i] = np.average(heights_in_bin, weights=weights)
#             else:
#                 binned_heights[i] = np.nan

#         # Remove NaN values for interpolation
#         valid = ~np.isnan(binned_heights)
#         bin_centers = bin_centers[valid]
#         binned_heights = binned_heights[valid]

#         if len(bin_centers) < 2:  # Ensure there are enough points for interpolation
#             print(f"   > Not enough points for interpolation for {pair_name}.")
#             continue

#         # Perform cubic spline interpolation
#         interpolated_lon = np.linspace(bin_centers[0], bin_centers[-1],
#                                        int(len(bin_centers) * adaptive_resolution_factor))
#         spline = CubicSpline(bin_centers, binned_heights)
#         interpolated_height = spline(interpolated_lon)

#         # Store the results
#         interpolated_profiles[pair_name]["lon"] = interpolated_lon
#         interpolated_profiles[pair_name]["height"] = interpolated_height

#     print("   > Interpolation completed.")
#     return interpolated_profiles
#     bin_size = 0.0001  # Bin size in longitude degrees
#     adaptive_resolution_factor = 50  # Resolution factor for interpolation
#     interpolated_combined_profiles = interpolate_combined_photon_profiles(classified_photons, bin_size, adaptive_resolution_factor)

from scipy.interpolate import PchipInterpolator

from scipy.ndimage import gaussian_filter1d

def interpolate_combined_photon_profiles(classified_photons, bin_size, adaptive_resolution_factor, smoothing_sigma=1):
    """
    Robustly interpolates the photon profiles for two combined datasets:
    - Top and Lower photons.
    - Top and Upper photons.

    This method applies:
    1. Gaussian-weighted smoothing to remove noise.
    2. PCHIP interpolation for monotonic piecewise cubic interpolation.

    Parameters:
        classified_photons (dict): Dictionary containing classified photons.
        bin_size (float): Bin size for grouping photons along the longitude axis.
        adaptive_resolution_factor (float): Factor to adjust the resolution of interpolation dynamically.
        smoothing_sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        dict: Dictionary containing the robust interpolated curves:
              Example: {
                  "top_lower": {"lon": [...], "height": [...]},
                  "top_upper": {"lon": [...], "height": [...]}
              }
    """
    print("13) Interpolating combined photon profiles with robust methods...")

    interpolated_profiles = {"top_lower": {"lon": [], "height": []},
                             "top_upper": {"lon": [], "height": []}}

    for pair_name, groups in [("top_lower", ["top", "lower"]), ("top_upper", ["top", "upper"])]:
        # Combine data from the specified groups
        combined_lon = np.concatenate([classified_photons[groups[0]]["lon"], classified_photons[groups[1]]["lon"]])
        combined_height = np.concatenate([classified_photons[groups[0]]["height"], classified_photons[groups[1]]["height"]])
        combined_density = np.concatenate([classified_photons[groups[0]]["density"], classified_photons[groups[1]]["density"]])

        if len(combined_lon) == 0:
            print(f"   > No data points available for {pair_name} interpolation.")
            continue

        # Bin the data along longitude
        bins = np.arange(np.min(combined_lon), np.max(combined_lon) + bin_size, bin_size)
        bin_indices = np.digitize(combined_lon, bins)

        # Compute median-density-weighted average heights per bin
        bin_centers = (bins[:-1] + bins[1:]) / 2
        binned_heights = np.zeros(len(bin_centers))

        for i in range(len(bin_centers)):
            in_bin = bin_indices == (i + 1)
            if np.any(in_bin):
                # Compute median density for the current bin
                median_density = np.median(combined_density[in_bin])

                # Apply median density as the weight for height averaging
                heights_in_bin = combined_height[in_bin]

                # Compute weighted average using the median density
                weights = np.full(len(heights_in_bin), median_density)
                binned_heights[i] = np.average(heights_in_bin, weights=weights)
            else:
                binned_heights[i] = np.nan

        # Remove NaN values for interpolation
        valid = ~np.isnan(binned_heights)
        bin_centers = bin_centers[valid]
        binned_heights = binned_heights[valid]

        if len(bin_centers) < 2:  # Ensure there are enough points for interpolation
            print(f"   > Not enough points for interpolation for {pair_name}.")
            continue

        # Apply Gaussian Smoothing to Reduce Noise
        smoothed_heights = gaussian_filter1d(binned_heights, sigma=smoothing_sigma)

        # Use PCHIP Interpolation (Monotonic Cubic Interpolation)
        interpolated_lon = np.linspace(bin_centers[0], bin_centers[-1],
                                       int(len(bin_centers) * adaptive_resolution_factor))
        pchip = PchipInterpolator(bin_centers, smoothed_heights)
        interpolated_height = pchip(interpolated_lon)

        # Store the results
        interpolated_profiles[pair_name]["lon"] = interpolated_lon
        interpolated_profiles[pair_name]["height"] = interpolated_height

    print("   > Interpolation completed with robust smoothing and PCHIP.")
    return interpolated_profiles

# Interpolate profiles
bin_size = 0.0001  # Bin size in longitude degrees
adaptive_resolution_factor = 50  # Resolution factor for interpolation
smoothing_sigma = 2  # Smoothing level for noise reduction

interpolated_combined_profiles = interpolate_combined_photon_profiles(
    classified_photons, bin_size, adaptive_resolution_factor, smoothing_sigma
)


# Plot the interpolated profiles
plt.figure(figsize=(12, 6))
plt.scatter(classified_photons["top"]["lon"], classified_photons["top"]["height"], c='green', s=5, alpha=0.3, label='Top Photons')
plt.scatter(classified_photons["lower"]["lon"], classified_photons["lower"]["height"], c='red', s=5, alpha=0.3, label='Lower Photons')
plt.scatter(classified_photons["upper"]["lon"], classified_photons["upper"]["height"], c='blue', s=5, alpha=0.3, label='Upper Photons')

# Plot interpolated curves
plt.plot(interpolated_combined_profiles["top_lower"]["lon"], interpolated_combined_profiles["top_lower"]["height"],
          color='yellow', label='Interpolated Bottom Surface Curve', linewidth=1.5)
plt.plot(interpolated_combined_profiles["top_upper"]["lon"], interpolated_combined_profiles["top_upper"]["height"],
          color='orange', label='Interpolated Top Surface Curve', linewidth=1.5)

plt.title("Interpolated Photon Profiles: Top Surface and Bottom Surface")
plt.xlabel("Longitude (°)")
plt.ylabel("Photon Height (m)")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


def compute_interpolated_line_distance(interpolated_profiles):
    """
    Computes the vertical distance between two interpolated lines ("top_lower" and "top_upper")
    along the track distance.

    Parameters:
        interpolated_profiles (dict): Dictionary containing the interpolated profiles for two lines:
                                      Example: {
                                          "top_lower": {"lon": [...], "height": [...]},
                                          "top_upper": {"lon": [...], "height": [...]}
                                      }

    Returns:
        dict: Dictionary containing:
              - "longitude": Array of longitude values where the lines are compared.
              - "distance": Array of absolute vertical distances between the two lines.
    """
    print("14) Computing correlation (distance) between two interpolated lines...")

    # Extract interpolated profiles
    top_lower = interpolated_profiles.get("top_lower", {})
    top_upper = interpolated_profiles.get("top_upper", {})

    if not top_lower or not top_upper:
        print("   > Missing one or both profiles for comparison.")
        return {"longitude": [], "distance": []}

    # Ensure the longitude values are the same for both lines
    lon_top_lower = np.array(top_lower["lon"])
    lon_top_upper = np.array(top_upper["lon"])

    if not np.array_equal(lon_top_lower, lon_top_upper):
        print("   > Longitude grids are not identical; interpolating to match...")
        # Interpolate the heights of one line onto the other longitude grid
        interp_top_upper = interp1d(lon_top_upper, top_upper["height"], kind='linear', fill_value="extrapolate")
        heights_top_upper = interp_top_upper(lon_top_lower)
        heights_top_lower = np.array(top_lower["height"])
        longitude = lon_top_lower
    else:
        # Use heights directly if the longitude grids are identical
        longitude = lon_top_lower
        heights_top_lower = np.array(top_lower["height"])
        heights_top_upper = np.array(top_upper["height"])

    # Compute absolute vertical distance between the two lines
    distance = np.abs(heights_top_upper - heights_top_lower)

    print("   > Correlation computation completed.")
    return {"longitude": longitude, "distance": distance}

# Compute distances between the interpolated lines
line_distance = compute_interpolated_line_distance(interpolated_combined_profiles)

# Plot the distance along the track
plt.figure(figsize=(12, 6))
plt.plot(line_distance["longitude"], line_distance["distance"], color='purple', linewidth=2)
plt.title("Vertical Distance Between Top and Bottom Surface Interpolated Lines")
plt.xlabel("Longitude (°)")
plt.ylabel("Vertical Distance (m)")
plt.grid(True)
plt.show()


# Function for the Haversine Formula
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculates the great-circle distance between two points on the Earth using the Haversine formula.

    Parameters:
        lon1, lat1 (float or numpy.ndarray): Longitude and latitude of the first point (in degrees).
        lon2, lat2 (float or numpy.ndarray): Longitude and latitude of the second point (in degrees).

    Returns:
        float or numpy.ndarray: Distance between the two points in meters.
    """
    # Earth's radius in meters
    R = 6378137  # WGS-84 Earth radius in meters

    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in meters
    distance = R * c

    return distance


def identify_melt_ponds_from_distance(line_distance, photon_lon_signal, photon_lat_signal):
    """
    Identifies melt ponds based on vertical distance thresholds and computes their properties:
    - Peak depth
    - Average depth
    - Width in meters
    - Position (latitude and longitude) of the average depth

    Parameters:
        line_distance (dict): Dictionary containing:
                              - "longitude": Longitude values where distances are computed.
                              - "distance": Vertical distances between two interpolated lines.
        photon_lon_signal (numpy.ndarray): Longitudes of all signal photons.
        photon_lat_signal (numpy.ndarray): Latitudes of all signal photons.

    Returns:
        list of dict: List of detected melt ponds with their properties:
                      Example: [{"peak_depth": float, "avg_depth": float,
                                 "width": float, "position": {"lat": float, "lon": float}}]
    """
    print("15) Identifying melt ponds based on vertical distance...")

    melt_ponds = []
    in_pond = False
    pond_start_idx = None

    # Iterate through the vertical distances
    for i, distance in enumerate(line_distance["distance"]):
        if distance >= 0.5:  # Threshold for melt pond detection
            if not in_pond:
                # Start a new pond
                in_pond = True
                pond_start_idx = i
        else:
            if in_pond:
                # End the current pond and calculate properties
                in_pond = False
                pond_end_idx = i - 1

                # Compute properties of the detected melt pond
                pond_longitudes = line_distance["longitude"][pond_start_idx:pond_end_idx + 1]
                pond_distances = line_distance["distance"][pond_start_idx:pond_end_idx + 1]

                peak_depth = np.max(pond_distances)
                avg_depth = np.mean(pond_distances)

                # Calculate the width in meters using Haversine distance
                start_lon, end_lon = pond_longitudes[0], pond_longitudes[-1]
                start_lat = photon_lat_signal[np.argmin(np.abs(photon_lon_signal - start_lon))]
                end_lat = photon_lat_signal[np.argmin(np.abs(photon_lon_signal - end_lon))]
                width = haversine_distance(start_lon, start_lat, end_lon, end_lat)

                # Compute the position of the average depth
                avg_lon = np.mean(pond_longitudes)
                avg_lat_idx = np.argmin(np.abs(photon_lon_signal - avg_lon))
                avg_lat = photon_lat_signal[avg_lat_idx]

                # Store the melt pond properties
                melt_ponds.append({
                    "peak_depth": peak_depth,
                    "avg_depth": avg_depth,
                    "width": width,
                    "position": {"lat": avg_lat, "lon": avg_lon}
                })

    # Handle the last pond if it ends at the end of the array
    if in_pond:
        pond_longitudes = line_distance["longitude"][pond_start_idx:]
        pond_distances = line_distance["distance"][pond_start_idx:]

        peak_depth = np.max(pond_distances)
        avg_depth = np.mean(pond_distances)

        # Calculate the width in meters using Haversine distance
        start_lon, end_lon = pond_longitudes[0], pond_longitudes[-1]
        start_lat = photon_lat_signal[np.argmin(np.abs(photon_lon_signal - start_lon))]
        end_lat = photon_lat_signal[np.argmin(np.abs(photon_lon_signal - end_lon))]
        width = haversine_distance(start_lon, start_lat, end_lon, end_lat)

        avg_lon = np.mean(pond_longitudes)
        avg_lat_idx = np.argmin(np.abs(photon_lon_signal - avg_lon))
        avg_lat = photon_lat_signal[avg_lat_idx]

        melt_ponds.append({
            "peak_depth": peak_depth,
            "avg_depth": avg_depth,
            "width": width,
            "position": {"lat": avg_lat, "lon": avg_lon}
        })

    print(f"   > Detected {len(melt_ponds)} melt pond(s).")
    return melt_ponds

# Example: Compute melt ponds
melt_ponds = identify_melt_ponds_from_distance(
    line_distance, photon_lon_signal, photon_lat_signal
)

# Display detected melt ponds
for i, pond in enumerate(melt_ponds, 1):
    print(f"Melt Pond {i}:")
    print(f"  Peak Depth: {pond['peak_depth']:.2f} m")
    print(f"  Average Depth: {pond['avg_depth']:.2f} m")
    print(f"  Width: {pond['width']:.2f} meters")
    print(f"  Position: Latitude = {pond['position']['lat']:.5f}, Longitude = {pond['position']['lon']:.5f}")
#%%


# FIFTH CELL: Mapping melt ponds

#%%


def plot_ground_track_and_melt_ponds(beam, full_track_lon, full_track_lat, photon_lon_signal, photon_lat_signal, melt_ponds):
    """
    Plots the entire ground track of the selected beam and the filtered signal photons,
    ensuring longitude continuity across the -180° to 180° boundary. Also includes a legend for detected melt ponds.
    The land is displayed in green and the ocean in light blue.

    Parameters:
        beam (str): Selected Beam for the ground-track.
        full_track_lon (numpy.ndarray): Full track longitude.
        full_track_lat (numpy.ndarray): Full track latitude.
        photon_lon_signal (numpy.ndarray): Longitudes of the selected signal photons.
        photon_lat_signal (numpy.ndarray): Latitudes of the selected signal photons.
        melt_ponds (list of dict): List of detected melt ponds with position (latitude & longitude).

    Returns:
        None: Displays the map.
    """
    print(f"Plotting ground track and melt pond positions for beam '{beam}'...")

    # **Ensure Longitude Continuity** (Correct Jumps at -180° / 180°)
    full_track_lon = np.unwrap(np.radians(full_track_lon), discont=np.pi)  # Convert to radians, unwrap
    full_track_lon = np.degrees(full_track_lon)  # Convert back to degrees

    photon_lon_signal = np.unwrap(np.radians(photon_lon_signal), discont=np.pi)
    photon_lon_signal = np.degrees(photon_lon_signal)

    # **Downsampling the Full Track for Efficiency**
    downsample_factor = 10000  # Adjust downsampling to prevent slow plotting
    full_track_lon = full_track_lon[::downsample_factor]
    full_track_lat = full_track_lat[::downsample_factor]

    # **Set up a polar stereographic projection for the North Pole**
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo()}, figsize=(10, 10))

    # **Set water background color to light blue**
    ax.set_facecolor("lightblue")

    # **Add features: coastlines and land**
    ax.add_feature(cfeature.LAND, facecolor='green')  # Land in green
    ax.coastlines(resolution='50m', color='black', linewidth=1)

    # **Plot the entire ground track (raw photons)**
    ax.plot(full_track_lon, full_track_lat, transform=ccrs.PlateCarree(),
            color='orange', linewidth=1.5, alpha=0.5, label=f'{beam} Full Track')

    # **Plot the selected signal photons (processed data)**
    ax.plot(photon_lon_signal, photon_lat_signal, transform=ccrs.PlateCarree(),
            color='blue', linewidth=2.5, label=f'{beam} Signal Photons')

    # **Add detected melt pond positions as markers**
    if melt_ponds:
        pond_lons = [pond["position"]["lon"] for pond in melt_ponds]
        pond_lats = [pond["position"]["lat"] for pond in melt_ponds]

        # Scatter plot for all melt ponds
        ax.scatter(pond_lons, pond_lats, transform=ccrs.PlateCarree(),
                   color='red', marker='x', s=200, label="Detected Melt Ponds")

        # Annotate first 10 melt ponds for clarity
        for pond in melt_ponds[:10]:  # Limit to first 10 annotations to reduce clutter
            lon, lat = pond["position"]["lon"], pond["position"]["lat"]
            ax.annotate(f"({lat:.2f}°, {lon:.2f}°)", xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        xytext=(5, 5), textcoords="offset points", fontsize=8, color='black')

    # **Add gridlines**
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()  # Format longitude labels
    gl.yformatter = LatitudeFormatter()  # Format latitude labels
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # **Set map extent to focus on the North Pole region**
    ax.set_extent([-180, 180, 70, 90], crs=ccrs.PlateCarree())

    # **Add a legend**
    ax.legend(loc='upper left', fontsize=10)

    # **Add title**
    plt.title(f"{beam} Ground Track and Melt Pond Positions (North Pole)", fontsize=14)

    # **Show the map**
    plt.show()

# Example usage
plot_ground_track_and_melt_ponds(beam, full_track_lon, full_track_lat, photon_lon_signal, photon_lat_signal, melt_ponds)

# Execution Time Reporting
end_time = time.time()
total_time = end_time - start_time

print(f"Processing complete. Total execution time: {total_time:.2f} seconds.")
#%%
