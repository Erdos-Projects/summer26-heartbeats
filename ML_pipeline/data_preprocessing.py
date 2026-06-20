"""
Preprocessing and feature extraction for the PAMAP2 activity data.

Feature definitions follow Yang et al., "Comparing Cross-Subject Performance on Human
Activities Recognition Using Learning Models".
The hand-crafted time/frequency features are listed in its Table 3, and the extraction
details are in its
Section III-C. Channel validity for PAMAP2 follows the dataset readme, Reiss & Stricker.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from itertools import groupby, combinations
from operator import itemgetter
from functools import partial

# Reuse the PAMAP2 column headers from read_data rather than restating them.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis_and_validation'))
from read_data import get_pamap2_headers

PAMAP2_HEADERS = get_pamap2_headers()
BODY_PARTS = ['hand', 'chest', 'ankle']

# Channels kept for feature extraction. Per the PAMAP2 readme, the
# orientation columns are invalid, the +-6g accelerometer can saturate, and temperature
# does not track motion, so all three are dropped. This leaves the (x/y/z)
# acc16, gyro, and mag on each IMU, plus heart rate.
TRIAXIAL_SENSORS = ['acc16', 'gyro', 'mag']
SENSOR_TRIADS = {
    f"{part}_{sensor}": [f"{part}_{sensor}_{axis}" for axis in ('x', 'y', 'z')]
    for part in BODY_PARTS for sensor in TRIAXIAL_SENSORS
}
# Heart rate is kept as a feature channel; its low sampling rate is not yet handled.
FEATURE_CHANNELS = ['heart_rate'] + [axis for triad in SENSOR_TRIADS.values() for axis in triad]

# Readable names for the agg outputs; any function not listed keeps the name agg gives it.
FEATURE_NAMES = {
    '_harmonic_mean': 'hmean',
    '_peak2peak_amp': 'p2p',
    '_sum_of_area': 'sum_abs',
    '_signal_mean_energy': 'mean_energy',
    'median_abs_deviation': 'mad',
}


class HeartbeatDataProcessor:
    def __init__(self, folder_path, filtered_df_path,window_size=2, step_size=1,boundary_cut=5,max_interpLength=0.1,verbose=True):
            """
            Initializes the data pipeline reader.
            
            Parameters:
            - file_path (str): Path to the df_intervals CSV data.
            - window_size (int): Number of consecutive intervals per training chunk.
            - step_size (int): How far the window slides forward (enables overlapping).
            - max_interpLength (flt): max length of NaNs in seconds to interpolate
            - verbose (bool): toggle print statements
            """
            self.folder_path =   folder_path
            self.filtered_df_path = filtered_df_path
            self.window_size = window_size
            self.step_size = step_size
            self.max_interpLength = max_interpLength
            self.boundary_cut = boundary_cut
            self.verbose = verbose
            # Initialize internal storage and stateful scaler
            self.df_filtered = None
            self.filtered_index = None
            self.scaler = StandardScaler()
            self.subject_segment_dict = {}
            self.features_df = None

    def _load_filtered_df(self,subject_num):
        #the data here already filtered by activity id!=0 and length>20
        filtered_df_path = f"{self.filtered_df_path}filtered_activities.csv"
        self.filtered_index = pd.read_csv(filtered_df_path)

    def preprocess_subjects(self,subject_array):
        # 1. Ensure the filtered index is loaded
        if self.filtered_index is None:
            self._load_filtered_df(subject_array)

        # 2. Preprocess each subject
        for subject_num in subject_array:
            file_path = f"{self.folder_path}subject{subject_num}.dat"
            df_raw = pd.read_csv(file_path, sep=r'\s+', header=None)

            subject_intervals = self.filtered_index[self.filtered_index['subject_id'] == subject_num]

            #interplote code should be a separate function under class and should be applied here on df_raw
            df_raw = self._interpolate_df(df_raw)
            #filter function should be a separate function under class and should be applied here after interplote on df_raw
            df_raw = self._filter_df(df_raw)

            # Name the columns now that interpolation and filtering, which work on the raw
            # integer columns, are done.
            df_raw.columns = PAMAP2_HEADERS

            #segment by window_size and step_size
            self.subject_segment_dict[subject_num] = []

            for interval_id, interval in subject_intervals.iterrows():
                start_t = int(interval['t1'])+self.boundary_cut
                end_t = int(interval['t2'])-self.boundary_cut

                #segment by window_size and step_size
                start_dt_array = np.arange(start_t,end_t-self.window_size,self.step_size)

                for start_dt in start_dt_array:
                    # 3. Extract the interval from the raw data
                    chunk = df_raw[(df_raw['timestamp']>=start_dt) & (df_raw['timestamp']<=start_dt+self.window_size)].copy()
                    if not chunk.empty:
                        chunk['subject_id'] = subject_num
                        chunk['interval_id'] = interval_id

                    self.subject_segment_dict[subject_num].append(chunk)
            #print(self.subject_segment_dict[subject_num][0])
            if self.verbose:
                print("successfully loaded subject",subject_num)
        # 4. Combine all chunks into a single DataFrame
        self.df_filtered  = pd.concat(self.subject_segment_dict[subject_num],ignore_index=True)

        # 5. Standardize the features
        # df_combined[['x', 'y', 'z']] = self.scaler.fit_transform(df_combined[['x', 'y', 'z']])

    # Remaining work, roughly by urgency:
    # 1. Check the feature values against the paper's Table 3 (not yet verified numerically).
    # 2. Handle NaNs. The features rely on agg skipping them, so a longer interpolation
    #    window may be needed.
    # 3. Add the frequency-domain streams. Section III-C also extracts Table 3 from the
    #    amplitude M=sqrt(x^2+y^2+z^2) and from STFT spectra; only the time domain is done.
    def extract_all_features(self):
        """
        Run extract_features on every window in subject_segment_dict and stack the rows into
        a feature matrix. Each row is one window, tagged with subject_id, interval_id, and
        activity_id. The matrix is returned and stored on self.features_df.
        """
        feature_rows = []
        for segments in self.subject_segment_dict.values():
            for segment in segments:
                # Some chunks appended by preprocess_subjects are empty, so skip them.
                if segment.empty:
                    continue
                row = self.extract_features(segment)
                # Each interval holds a single activity, so activity_id is constant within
                # a window and the first sample's value labels the whole window.
                row['subject_id'] = segment['subject_id'].iloc[0]
                row['interval_id'] = segment['interval_id'].iloc[0]
                row['activity_id'] = segment['activity_id'].iloc[0]
                feature_rows.append(row)

        self.features_df = pd.DataFrame(feature_rows).reset_index(drop=True)
        if self.verbose:
            print("extracted features for", len(self.features_df), "windows")
        return self.features_df

    def extract_features(self, window_df):
        """
        Extract one row of features from a single window of samples.

        Parameters:
        - window_df (pd.DataFrame): samples for one window, with named PAMAP2 columns.

        Returns a pandas Series indexed by "<channel>_<feature>", covering the
        per-axis statistics from the paper's Table 3 plus the within-sensor axis
        correlations.
        """
        # Restrict to the modelled channels. The metadata columns (timestamp and the ids)
        # and the dropped sensors (orientation, the +-6g accelerometer, and temperature)
        # are ignored here.
        channels = window_df[FEATURE_CHANNELS]

        # The custom callables below cover features with no direct pandas/scipy name.
        # They take a single column (a Series) because that is what agg hands them.
        def _peak2peak_amp(column):
            # Peak-to-peak amplitude is the largest sample minus the smallest (Table 3).
            return column.max() - column.min()

        def _sum_of_area(column):
            # Sum of area is the total of the absolute sample values (Table 3).
            return column.abs().sum()

        def _signal_mean_energy(column):
            # Signal mean energy is the mean of the squared samples (Table 3).
            return (column ** 2).mean()

        def _harmonic_mean(column):
            # Harmonic mean n/sum(1/x) on the signed samples (Table 3; Section III-C defines
            # x as one axis's samples in a window). This is not scipy.stats.hmean, which
            # returns NaN on signed input. Zero division is set to 0, per Section III-C.
            column = column.dropna()
            n = len(column)
            if n == 0:
                return np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = (1.0 / column).sum()
            if not np.isfinite(denominator) or denominator == 0:
                return 0.0
            harmonic_mean = n / denominator
            return harmonic_mean if np.isfinite(harmonic_mean) else 0.0

        # agg accepts three kinds of function: strings name pandas methods, the stats.*
        # entries are scipy functions, and the underscore callables above are custom. Each
        # function is applied to every channel at once, giving a (feature x channel) table.
        stats_df = channels.agg(func=[
            'mean',
            _harmonic_mean,
            'std',
            'max',
            'min',
            _peak2peak_amp,
            'median',
            partial(stats.median_abs_deviation, nan_policy='omit'),
            partial(stats.iqr, nan_policy='omit'),
            _sum_of_area,
            _signal_mean_energy,
            'skew',
            'kurtosis',
        ])

        # Give the rows readable names, then flatten the (feature x channel) table into a
        # single row indexed "<channel>_<feature>", for example hand_acc16_x_mean.
        stats_df = stats_df.rename(index=FEATURE_NAMES)
        flat = stats_df.stack()
        flat.index = [f"{channel}_{feature}" for feature, channel in flat.index]

        # Add the within-sensor axis correlations (Pearson).
        correlations = self._axis_correlations(channels)

        return pd.concat([flat, correlations])

    def _axis_correlations(self, channels):
        """
        Pearson correlation between axis pairs within each triaxial sensor.

        For every sensor (e.g. hand_acc16) the three axis pairs (x,y), (x,z), (y,z)
        are correlated. Correlations are not taken across different sensors or
        different IMUs. A constant (zero-variance) axis gives an undefined
        correlation, which the paper sets to 0.

        Returns a pandas Series indexed by "<sensor>_corr_<pair>" (e.g.
        hand_acc16_corr_xy).
        """
        correlations = {}
        for sensor, axes in SENSOR_TRIADS.items():
            for axis_a, axis_b in combinations(axes, 2):
                correlation = channels[axis_a].corr(channels[axis_b])
                # A constant (zero-variance) axis makes the correlation undefined (NaN),
                # which the paper sets to 0.
                if pd.isna(correlation):
                    correlation = 0.0
                # Name each correlation by the trailing axis letters, so the (x, y) pair
                # becomes "..._corr_xy".
                correlations[f"{sensor}_corr_{axis_a[-1]}{axis_b[-1]}"] = correlation
        return pd.Series(correlations)

    def _interpolate_df(self,df_raw):

        columns = list(df_raw.columns.values)

        # we will want to deal with heart rate sampling as a special case, since it is sampled at a low frequency
        # skip for now, implement here or elsewhere when we decide how to proceed

        columns.remove(0)
        columns.remove(1)
        columns.remove(2)

        for column in columns:

            # find which rows are NaN
            null_search = df_raw[column].isnull()
            null_list = null_search[null_search].index.values

            if len(null_list) != 0:
                i = null_list[0]
                # group NaN rows by consecutive sequences
                for k, g in groupby(enumerate(null_list), lambda x: x[0]-x[1]):
                    n_set = list(map(itemgetter(1), g))
                    # skip NaNs at the beginning or end of sequence
                    if n_set[0] == df_raw.index.values[0] or n_set[-1] == df_raw.index.values[-1]:
                        pass
                    elif df_raw[0][n_set[-1]] - df_raw[0][n_set[0]] < self.max_interpLength:
                        #get values for points before and after missing data
                        'TODO: benchmark performance vs pandas interp method'
                        n1 = n_set[0]-1
                        n2 = n_set[-1]+1
                        t1 = df_raw[0][n1]
                        t2 = df_raw[0][n2]
                        y1 = df_raw[column][n1]
                        y2 = df_raw[column][n2]
                        #slope for interpolation
                        m = (y2-y1)/(t2-t1)

                        # calculate and insert interpolated values
                        tvals = np.array(df_raw.loc[n_set][0].values)
                        df_raw.loc[n_set,column] = y1 + m*(tvals - t1)

        return df_raw

    def _filter_df(self,df_raw):
            #filter function should be a separate function under class and should be applied here after interplote on df_raw
            return df_raw