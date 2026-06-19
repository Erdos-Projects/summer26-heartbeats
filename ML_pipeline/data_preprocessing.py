from tracemalloc import start
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import groupby
from operator import itemgetter

class HeartbeatDataProcessor:
    def __init__(self, folder_path, filtered_df_path,window_size=2, step_size=1,boundary_cut=5,interp_limit=10,verbose=True):
            """
            Initializes the data pipeline reader.
            
            Parameters:
            - file_path (str): Path to the df_intervals CSV data.
            - window_size (int): Number of consecutive intervals per training chunk.
            - step_size (int): How far the window slides forward (enables overlapping).
            - interp_limit (int): max number of consecutive NaNs to interpolate, default 10 (0.1s at 100Hz)
            - verbose (bool): toggle print statements
            """
            self.folder_path =   folder_path
            self.filtered_df_path = filtered_df_path
            self.window_size = window_size
            self.step_size = step_size
            self.interp_limit = interp_limit
            self.boundary_cut = boundary_cut
            self.verbose = verbose
            # Initialize internal storage and stateful scaler
            self.df_filtered = None
            self.filtered_index = None
            self.scaler = StandardScaler()
            self.subject_segment_dict = {}

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
            #setting index to timestamp!
            df_raw.set_index(0)

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
                    chunk = df_raw[(df_raw[0]>=start_dt) & (df_raw[0]<=start_dt+self.window_size)].copy()
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

    def extract_features(self,window_df):
            # Extract features from the windowed data
            return self

    def _interpolate_df(self,df_raw):

        columns = list(df_raw.columns.values)

        debug_print = False
        

        # we will want to deal with heart rate sampling as a special case, since it is sampled at a low frequency
        # skip for now, implement here or elsewhere when we decide how to proceed

        columns.remove(0)
        columns.remove(1)
        columns.remove(2)

        # I think the pandas interp can do sets of columns simultaneously, but leaving the loop for now as I do some testing
        'TODO: consider replacing loop with simultaneous interpolation for a cleaner look'
        for column in columns:

            # find which rows are NaN
            missing_pct = df_raw[column].isnull().mean() * 100
            if debug_print:
                 print('Column '+str(column)+' has '+str(round(missing_pct,4))+'% NaNs.')
                 print('linear interp...')

            df_raw[column] = df_raw[column].interpolate(method='index',limit=self.interp_limit)

            missing_pct = df_raw[column].isnull().mean() * 100
            if debug_print:
                 print('Column '+str(column)+' has '+str(round(missing_pct,4))+'% NaNs.\n')

            'TODO: test quadratic interpolation for larger gaps'
            # if self.verbose:
            #      print('quadratic interp...')

            # df_raw[column] = df_raw[column].interpolate(method='index',limit=100)

            # missing_pct = df_raw[column].isnull().mean() * 100
            # if self.verbose:
            #      print('Column '+str(column)+' has '+str(round(missing_pct,4))+'% NaNs.\n')

        return df_raw

    def _filter_df(self,df_raw):
            #filter function should be a separate function under class and should be applied here after interplote on df_raw
            return df_raw