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
from scipy.signal import stft
from itertools import groupby, combinations
from operator import itemgetter
from functools import partial

# Reuse the PAMAP2 column headers from read_data rather than restating them.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis_and_validation'))
from read_data import get_pamap2_headers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_filtering import HeartRateFilter


class HeartbeatDataProcessor:
    def __init__(self, folder_path, filtered_df_path,window_size=2, step_size=1,boundary_cut=5,interp_limit=10,sample_rate=100,stft_nperseg=64,stft_noverlap=32,stft_window='hann',include_amplitude=True,include_frequency=True,verbose=True):
            """
            Initializes the data pipeline reader.

            Parameters:
            - file_path (str): Path to the df_intervals CSV data.
            - window_size (int): Number of consecutive intervals per training chunk.
            - step_size (int): How far the window slides forward (enables overlapping).
            - interp_limit (int): max number of consecutive NaNs to interpolate, default 10 (0.1s at 100Hz)
            - sample_rate (int): sample rate in Hz, used for the frequency-domain STFT.
            - stft_nperseg (int): STFT segment length in samples. The paper leaves it
              unspecified, so this is our choice (clamped to the window length per signal).
            - stft_noverlap (int): STFT segment overlap in samples, likewise our choice.
            - stft_window (str): STFT window function passed to scipy.signal.stft.
            - include_amplitude (bool): add the per-sensor amplitude M = sqrt(x^2+y^2+z^2)
              time-domain features. Cheap, on by default.
            - include_frequency (bool): add the frequency-domain (STFT magnitude) features for
              both the axes and the amplitudes. This is the slow step (an STFT per signal per
              window). Set False for a fast time-domain-only run.
            - verbose (bool): toggle print statements
            """
            self.folder_path =   folder_path
            self.filtered_df_path = filtered_df_path
            self.window_size = window_size
            self.step_size = step_size
            self.interp_limit = interp_limit
            self.boundary_cut = boundary_cut
            self.sample_rate = sample_rate
            self.stft_nperseg = stft_nperseg
            self.stft_noverlap = stft_noverlap
            self.stft_window = stft_window
            self.include_amplitude = include_amplitude
            self.include_frequency = include_frequency
            self.verbose = verbose
            # Initialize internal storage. Feature scaling is intentionally not done here:
            # it must be fit on the training fold only (inside the model pipeline) to avoid
            # leakage, so the extractor returns raw, unscaled features.
            self.df_filtered = None
            self.filtered_index = None
            self.subject_segment_dict = {}
            self.features_df = None

            # PAMAP2 column headers, reused from read_data rather than restated.
            self.PAMAP2_HEADERS = get_pamap2_headers()
            self.BODY_PARTS = ['hand', 'chest', 'ankle']

            # Channels kept for feature extraction. Per the PAMAP2 readme, the
            # orientation columns, the +-6g accelerometer and temperature are dropped.
            self.TRIAXIAL_SENSORS = ['acc16', 'gyro', 'mag']
            self.SENSOR_TRIADS = {
                f"{part}_{sensor}": [f"{part}_{sensor}_{axis}" for axis in ('x', 'y', 'z')]
                for part in self.BODY_PARTS for sensor in self.TRIAXIAL_SENSORS
            }
            # The 27 triaxial motion axes, and the same plus heart rate as feature channels.
            self.MOTION_AXES = [
                axis for triad in self.SENSOR_TRIADS.values() for axis in triad
            ]
            self.FEATURE_CHANNELS = ['heart_rate'] + self.MOTION_AXES
            self.FEATURE_NAMES = {
                '_harmonic_mean': 'hmean',
                '_peak2peak_amp': 'p2p',
                '_sum_of_area': 'sum_abs',
                '_signal_mean_energy': 'mean_energy',
                'median_abs_deviation': 'mad',
                '_table3_std': 'table_3_std',
                '_table3_skew': 'table_3_skew',
                '_table3_kurtosis': 'table_3_kurtosis',
            }

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
            # Name the columns at read so every step downstream works by column name. This
            # keeps interpolation, filtering, and feature selection robust to columns being
            # dropped or reordered upstream.
            df_raw = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.PAMAP2_HEADERS)

            subject_intervals = self.filtered_index[self.filtered_index['subject_id'] == subject_num]

            #interplote code should be a separate function under class and should be applied here on df_raw
            df_raw = self._interpolate_df(df_raw)
            #filter function should be a separate function under class and should be applied here after interplote on df_raw
            df_raw = self._filter_df(df_raw)

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

    # Remaining work, roughly by urgency:
    # 1. Decide a NaN / zero-variance policy. The features rely on agg skipping NaNs, but a
    #    constant channel (e.g. heart rate over a short window) still gives NaN for the
    #    Table 3 skew and kurtosis, which scipy returns on zero variance while pandas
    #    returns 0. The paper sets only harmonic mean and Pearson to 0 on zero division.
    # 2. Decide whether to keep heart rate, which the paper excludes (it uses IMU data only).
    # 3. Confirm the STFT parameters (segment length, overlap, window), which the paper does
    #    not state. The current values in __init__ are our choice.
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
                activity_ids = segment['activity_id']
                assert activity_ids.nunique() == 1, (
                    f"window spans multiple activities {activity_ids.unique()} "
                    f"(subject {segment['subject_id'].iloc[0]}, "
                    f"interval {segment['interval_id'].iloc[0]})"
                )
                row['subject_id'] = segment['subject_id'].iloc[0]
                row['interval_id'] = segment['interval_id'].iloc[0]
                row['activity_id'] = activity_ids.iloc[0]
                feature_rows.append(row)

        self.features_df = pd.DataFrame(feature_rows).reset_index(drop=True)
        if self.verbose:
            print("extracted features for", len(self.features_df), "windows")
        return self.features_df

    def save_features(self, path):
        """
        Write the extracted feature matrix to CSV so the slow extraction only has to run
        once. extract_all_features must have been called first.
        """
        if self.features_df is None:
            raise RuntimeError("no features to save; run extract_all_features first")
        self.features_df.to_csv(path, index=False)
        if self.verbose:
            print("saved features to", path)

    def extract_features(self, window_df):
        """
        Extract one row of features from a single window of samples.

        Parameters:
        - window_df (pd.DataFrame): samples for one window, with named PAMAP2 columns.

        Returns a pandas Series indexed by "<channel>_<feature>". Following Section III-C,
        the Table 3 statistics are taken over up to four streams: the original time-domain
        channels (always), the per-sensor amplitude M in the time domain (if
        include_amplitude), and the STFT magnitude of both in the frequency domain (if
        include_frequency). Those channels carry a "_stft" suffix. The within-sensor axis
        correlations are added from the time domain. The returned index therefore depends on
        the include_amplitude / include_frequency settings.
        """
        # The custom functions below cover features with no direct pandas/scipy name.
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
            # Harmonic mean n/sum(1/x) on the signed samples (Table 3, Section III-C defines
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

        # Table 3 defines std, skew, and kurtosis differently from the pandas methods used
        # above, so the paper's forms are added as separate features rather than replacing
        # them. The differences are the divisor and the bias/Fisher conventions.
        def _table3_std(column):
            # Population standard deviation sqrt((1/n) sum (x - mean)^2), dividing by n. The
            # pandas 'std' above divides by n - 1.
            return column.std(ddof=0)

        def _table3_skew(column):
            # Biased (population) skewness. The pandas 'skew' above applies a sample bias
            # correction. A constant (zero-variance) window leaves skewness undefined, so it
            # is set to 0, matching how the paper handles the harmonic mean and Pearson zero
            # division, and matching what pandas 'skew' returns on constant input.
            column = column.dropna()
            if len(column) == 0:
                return np.nan
            if column.std(ddof=0) == 0:
                return 0.0
            return stats.skew(column, bias=True)

        def _table3_kurtosis(column):
            # Raw fourth standardized moment, for which a normal distribution gives 3. The
            # pandas 'kurtosis' above returns the Fisher (excess) value, normal 0, with a
            # bias correction. A constant window is set to 0 as for the skew above.
            column = column.dropna()
            if len(column) == 0:
                return np.nan
            if column.std(ddof=0) == 0:
                return 0.0
            return stats.kurtosis(column, fisher=False, bias=True)

        # agg accepts three kinds of function: strings name pandas methods, the stats.*
        # entries are scipy functions, and the underscore callables above are custom.
        agg_funcs = [
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
            _table3_std,
            _table3_skew,
            _table3_kurtosis,
        ]

        def _per_channel_features(signals):
            # Apply every function to every column, then flatten the (feature x channel)
            # table into one row indexed "<channel>_<feature>", e.g. hand_acc16_x_mean.
            table = signals.agg(func=agg_funcs).rename(index=self.FEATURE_NAMES)
            flat = table.stack()
            flat.index = [f"{channel}_{feature}" for feature, channel in flat.index]
            return flat

        # The streams of Section III-C. The original time domain is always extracted. The
        # amplitude and frequency-domain streams are optional (see include_amplitude /
        # include_frequency in __init__), since the STFT is the slow step. Heart rate is kept
        # only in the original time domain. It has no triad for an amplitude, and its sparse,
        # low-rate sampling makes its spectrum meaningless.
        channels = window_df[self.FEATURE_CHANNELS]
        feature_parts = [_per_channel_features(channels)]

        # Amplitude M per sensor feeds both the amplitude time-domain features and the
        # amplitude frequency-domain features, so compute it once if either stream is on.
        amplitude = None
        if self.include_amplitude or self.include_frequency:
            amplitude = self._amplitude_signals(window_df)
        if self.include_amplitude:
            feature_parts.append(_per_channel_features(amplitude))

        if self.include_frequency:
            axis_spectra = self._stft_magnitude_signals(window_df[self.MOTION_AXES])
            amplitude_spectra = self._stft_magnitude_signals(amplitude)
            feature_parts.append(_per_channel_features(axis_spectra))
            feature_parts.append(_per_channel_features(amplitude_spectra))

        # Axis correlations are taken in the time domain only.
        feature_parts.append(self._axis_correlations(channels))
        return pd.concat(feature_parts)

    def _amplitude_signals(self, window_df):
        """
        Amplitude M = sqrt(x^2 + y^2 + z^2) per triaxial sensor (Section III-C, eq. 3).

        Collapsing the three axes into one magnitude removes orientation dependence.
        Returns a DataFrame whose columns are "<sensor>_amp" (e.g. hand_acc16_amp), one
        per sensor.
        """
        amplitudes = {}
        for sensor, axes in self.SENSOR_TRIADS.items():
            amplitudes[f"{sensor}_amp"] = np.sqrt((window_df[axes] ** 2).sum(axis=1))
        return pd.DataFrame(amplitudes)

    def _stft_magnitude_signals(self, signals):
        """
        Short-time Fourier transform magnitude of each signal, flattened to a vector.

        Section III-C extracts the Table 3 features from the frequency domain as well,
        obtained by an STFT of the original and amplitude signals, but does not state the
        STFT parameters. The window, segment length, and overlap are therefore our choice
        (set in __init__), and the features are computed over the flattened magnitude
        spectrum. NaNs are linearly interpolated first, since the STFT does not skip them.

        Returns a DataFrame whose columns are the input columns suffixed with "_stft".
        """
        spectra = {}
        for column in signals.columns:
            signal = signals[column].interpolate().bfill().ffill().to_numpy()
            # nperseg cannot exceed the signal length, and noverlap must be below nperseg.
            nperseg = min(self.stft_nperseg, len(signal))
            noverlap = min(self.stft_noverlap, nperseg - 1) if nperseg > 1 else 0
            _, _, Zxx = stft(signal, fs=self.sample_rate, window=self.stft_window,
                             nperseg=nperseg, noverlap=noverlap)
            spectra[f"{column}_stft"] = np.abs(Zxx).ravel()
        return pd.DataFrame(spectra)

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
        for sensor, axes in self.SENSOR_TRIADS.items():
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

        # we will want to deal with heart rate sampling as a special case, since it is sampled at a low frequency
        # skip for now, implement here or elsewhere when we decide how to proceed

        # Interpolate every column except the metadata, identified by name (not position) so
        # that dropping or reordering columns upstream cannot break this.
        metadata = ['timestamp', 'activity_id', 'heart_rate']
        columns = [c for c in df_raw.columns if c not in metadata]

        #create mask of all entries NOT part of long nan sequences
        mask = df_raw[columns].copy()
        grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
        for i in columns:
            mask[i] = (grp.groupby(i).transform('size') <= self.interp_limit) | df_raw[i].notnull()

        missing_pct = df_raw[columns].isnull().to_numpy().flatten().mean() * 100
        if self.verbose:
            print('Selected DataFrame columns have '+str(round(missing_pct,4))+'% NaNs.')
            print('Interpolating selected columns...')

        #interpolate!
        df_raw[columns] = df_raw[columns].interpolate(method='index',limit=self.interp_limit,limit_area='inside')

        #backfill all values except long nan sequences identified by mask
        df_raw[columns] = df_raw[columns].bfill()[mask]

        missing_pct = df_raw[columns].isnull().to_numpy().flatten().mean() * 100
        if self.verbose:
            print('Selected DataFrame columns now have '+str(round(missing_pct,4))+'% NaNs!\n')

        'TODO: test quadratic interpolation for larger gaps'

        return df_raw

    def _filter_df(self, df_raw):
        # Low-pass filter the motion channels (paper, Section III-A). Heart rate and the
        # metadata columns are left untouched.
        signal_filter = HeartRateFilter(kernel_size=5, cutoff=11.0, fs=self.sample_rate, order=5)
        for column in self.MOTION_AXES:
            df_raw[column] = signal_filter.fit_transform(df_raw[column]).ravel()
        return df_raw