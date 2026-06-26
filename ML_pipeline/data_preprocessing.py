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
# Imported as scipy_stats so the extract_features "stats" argument cannot shadow it.
from scipy import stats as scipy_stats
from scipy.signal import stft
from itertools import groupby, combinations
from operator import itemgetter

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
            # The Section III-C feature streams that extract_features can select.
            self.STREAMS = ('time', 'amplitude', 'frequency')
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
    # 1. Decide whether to keep heart rate, which the paper excludes (it uses IMU data only).
    # 2. Confirm the STFT parameters (segment length, overlap, window), which the paper does
    #    not state. The current values in __init__ are our choice.
    #
    # NaN check: the feature matrix was 0 NaN on the fast path (time domain only) across all
    # 8 subjects (18,103 windows), and 0 NaN on the full path on subject 101. The full path
    # is NaN-free by construction, since its amplitude, STFT, and correlation features derive
    # from the motion channels, which the filter leaves NaN-free, but it was not run on every
    # subject.
    def extract_all_features(self, streams=None, channels=None, stats=None):
        """
        Run extract_features on every window in subject_segment_dict and stack the rows into
        a feature matrix. Each row is one window, tagged with subject_id, interval_id, and
        activity_id. The matrix is returned and stored on self.features_df.

        streams, channels, and stats select which features to compute and are passed straight
        through to extract_features. See that method. They default to the full feature set
        (the constructor's include_amplitude / include_frequency set the default streams).
        """
        feature_rows = []
        for segments in self.subject_segment_dict.values():
            for segment in segments:
                # Some chunks appended by preprocess_subjects are empty, so skip them.
                if segment.empty:
                    continue
                row = self.extract_features(segment, streams=streams,
                                            channels=channels, stats=stats)
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

    @staticmethod
    def _resolve_selection(requested, valid, kind):
        """
        Normalize a feature-selection argument. None means "all of valid". Otherwise the
        request must be a non-empty subset of valid. An unknown or empty request raises
        ValueError. The result keeps the canonical order of valid.
        """
        if requested is None:
            return list(valid)
        requested = list(requested)
        if not requested:
            raise ValueError(f"no {kind} selected; pass None for all or a non-empty subset")
        unknown = [r for r in requested if r not in valid]
        if unknown:
            raise ValueError(f"unknown {kind} {unknown}; valid options are {list(valid)}")
        chosen = set(requested)
        return [v for v in valid if v in chosen]

    def extract_features(self, window_df, streams=None, channels=None, stats=None):
        """
        Extract one row of features from a single window of samples.

        Parameters:
        - window_df (pd.DataFrame): samples for one window, with named PAMAP2 columns.
        - streams (list of str or None): which Section III-C streams to compute, any of
          'time', 'amplitude', 'frequency'. None uses the constructor's include_amplitude /
          include_frequency for the default ('time' is on by default).
        - channels (list of str or None): which feature channels to include, a subset of
          FEATURE_CHANNELS (heart rate and the 27 motion axes). None means all. Amplitude and
          axis-correlation features need a full triad, so they are produced only for sensors
          whose three axes are all selected.
        - stats (list of str or None): which Table 3 statistics to compute, named by their
          output suffix (e.g. 'mean', 'hmean', 'p2p', 'table_3_std'). None means all.

        An unknown or empty streams / channels / stats raises ValueError.

        Returns a pandas Series indexed by "<channel>_<feature>". Frequency-domain channels
        carry a "_stft" suffix. The returned index depends on the selection.
        """
        # Resolve the stream and channel selections (stats are resolved below, once the
        # statistic table is built). None means "all"; an unknown entry raises.
        if streams is None:
            streams = ['time']
            if self.include_amplitude:
                streams.append('amplitude')
            if self.include_frequency:
                streams.append('frequency')
        streams = self._resolve_selection(streams, self.STREAMS, 'stream')
        channels = self._resolve_selection(channels, self.FEATURE_CHANNELS, 'channel')

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

        # Named wrappers around the scipy statistics. They exist so the aggregation table
        # below can hold plain callables instead of functools.partial: a partial has no
        # __name__, and pandas inspects __name__ when classifying an aggregation, so the
        # partial form is fragile across pandas versions (see _per_channel_features).
        def _median_abs_deviation(column):
            return scipy_stats.median_abs_deviation(column, nan_policy='omit')

        def _interquartile_range(column):
            return scipy_stats.iqr(column, nan_policy='omit')

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
            return scipy_stats.skew(column, bias=True)

        def _table3_kurtosis(column):
            # Raw fourth standardized moment, for which a normal distribution gives 3. The
            # pandas 'kurtosis' above returns the Fisher (excess) value, normal 0, with a
            # bias correction. A constant window is set to 0 as for the skew above.
            column = column.dropna()
            if len(column) == 0:
                return np.nan
            if column.std(ddof=0) == 0:
                return 0.0
            return scipy_stats.kurtosis(column, fisher=False, bias=True)

        # Each (label, func) pair is one Table 3 statistic. func is either the name of a
        # pandas method (a string) or one of the callables above. The label is the row name
        # before FEATURE_NAMES renames it.
        agg_funcs = [
            ('mean', 'mean'),
            ('_harmonic_mean', _harmonic_mean),
            ('std', 'std'),
            ('max', 'max'),
            ('min', 'min'),
            ('_peak2peak_amp', _peak2peak_amp),
            ('median', 'median'),
            ('median_abs_deviation', _median_abs_deviation),
            ('iqr', _interquartile_range),
            ('_sum_of_area', _sum_of_area),
            ('_signal_mean_energy', _signal_mean_energy),
            ('skew', 'skew'),
            ('kurtosis', 'kurtosis'),
            ('_table3_std', _table3_std),
            ('_table3_skew', _table3_skew),
            ('_table3_kurtosis', _table3_kurtosis),
        ]

        # Resolve which statistics to compute. They are named by their output suffix (e.g.
        # "hmean", "p2p"). An unknown name raises in _resolve_selection.
        available_stats = [self.FEATURE_NAMES.get(label, label) for label, _ in agg_funcs]
        stat_names = set(self._resolve_selection(stats, available_stats, 'statistic'))
        selected_agg = [(label, func) for label, func in agg_funcs
                        if self.FEATURE_NAMES.get(label, label) in stat_names]

        def _per_channel_features(signals):
            # Apply every selected statistic to every column, then flatten the
            # (feature x channel) table into one row indexed "<channel>_<feature>", e.g.
            # hand_acc16_x_mean.
            #
            # Built explicitly instead of DataFrame.agg(list-of-funcs): pandas 2.x and 3.x
            # classify some callables differently in the list form (e.g. a function named
            # "iqr" is treated as a transform on 2.x), which raises "cannot combine transform
            # and aggregation operations" on 2.x. Calling each statistic directly avoids that.
            rows = {
                label: (getattr(signals, func)() if isinstance(func, str)
                        else signals.apply(func))
                for label, func in selected_agg
            }
            table = pd.DataFrame(rows).T.rename(index=self.FEATURE_NAMES)
            flat = table.stack()
            flat.index = [f"{channel}_{feature}" for feature, channel in flat.index]
            return flat

        # The channel groups the selected streams act on. motion_axes are the selected
        # triaxial axes (heart rate has no useful spectrum, so it is excluded). sensors are
        # the triads whose three axes are all selected, since an amplitude or an axis
        # correlation needs the full triad.
        selected = set(channels)
        motion_axes = [c for c in self.MOTION_AXES if c in selected]
        sensors = [s for s, axes in self.SENSOR_TRIADS.items()
                   if all(a in selected for a in axes)]

        # Assemble the requested streams. The time stream is the original channels plus the
        # within-sensor axis correlations. The amplitude stream is the per-sensor magnitude M.
        # The frequency stream is the STFT magnitude of the axes and of the amplitudes. The
        # amplitude signal feeds both the amplitude and frequency streams, so compute it once.
        feature_parts = []
        if 'time' in streams:
            feature_parts.append(_per_channel_features(window_df[channels]))

        amplitude = None
        if 'amplitude' in streams or 'frequency' in streams:
            amplitude = self._amplitude_signals(window_df, sensors)
        if 'amplitude' in streams and sensors:
            feature_parts.append(_per_channel_features(amplitude))

        if 'frequency' in streams:
            if motion_axes:
                axis_spectra = self._stft_magnitude_signals(window_df[motion_axes])
                feature_parts.append(_per_channel_features(axis_spectra))
            if sensors:
                amplitude_spectra = self._stft_magnitude_signals(amplitude)
                feature_parts.append(_per_channel_features(amplitude_spectra))

        # Axis correlations are taken in the time domain only.
        if 'time' in streams and sensors:
            feature_parts.append(self._axis_correlations(window_df[channels], sensors))

        if not feature_parts:
            raise ValueError("the requested streams and channels produced no features")
        return pd.concat(feature_parts)

    def _amplitude_signals(self, window_df, sensors=None):
        """
        Amplitude M = sqrt(x^2 + y^2 + z^2) per triaxial sensor (Section III-C, eq. 3).

        Collapsing the three axes into one magnitude removes orientation dependence.
        sensors restricts which triads are used (None means all). Returns a DataFrame whose
        columns are "<sensor>_amp" (e.g. hand_acc16_amp), one per sensor.
        """
        triads = self.SENSOR_TRIADS if sensors is None else {s: self.SENSOR_TRIADS[s] for s in sensors}
        amplitudes = {}
        for sensor, axes in triads.items():
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

    def _axis_correlations(self, channels, sensors=None):
        """
        Pearson correlation between axis pairs within each triaxial sensor.

        For every sensor (e.g. hand_acc16) the three axis pairs (x,y), (x,z), (y,z)
        are correlated. Correlations are not taken across different sensors or
        different IMUs. A constant (zero-variance) axis gives an undefined
        correlation, which the paper sets to 0. sensors restricts which triads are
        used (None means all).

        Returns a pandas Series indexed by "<sensor>_corr_<pair>" (e.g.
        hand_acc16_corr_xy).
        """
        triads = self.SENSOR_TRIADS if sensors is None else {s: self.SENSOR_TRIADS[s] for s in sensors}
        correlations = {}
        for sensor, axes in triads.items():
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