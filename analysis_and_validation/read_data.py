import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter

def get_activityMap():

    activity_map = {
        0: 'Transient/Break', 1: 'Lying down', 2: 'Sitting', 3: 'Standing', 
        4: 'Walking', 5: 'Running', 6: 'Cycling', 7: 'Nordic walking', 
        9: 'Watching TV', 10: 'Computer work', 11: 'Car driving', 
        12: 'Ascending stairs', 13: 'Descending stairs', 16: 'Vacuum cleaning', 
        17: 'Ironing', 18: 'Folding laundry', 19: 'House cleaning', 
        20: 'Playing soccer', 24: 'Rope jumping'
    }

    return activity_map

def get_pamap2_headers():
    headers = ['timestamp', 'activity_id', 'heart_rate']
    
    imu_features = [
        'temp', 
        'acc16_x', 'acc16_y', 'acc16_z', 
        'acc6_x', 'acc6_y', 'acc6_z', 
        'gyro_x', 'gyro_y', 'gyro_z', 
        'mag_x', 'mag_y', 'mag_z', 
        'orient_w', 'orient_x', 'orient_y', 'orient_z'
    ]
    
    for body_part in ['hand', 'chest', 'ankle']:
        for feature in imu_features:
            # Fixed the typo here!
            headers.append(f"{body_part}_{feature}")
            
    return headers

def load_subject(subject_num, folder_path='../data/PAMAP2_Dataset/Protocol/'):

    headers = get_pamap2_headers()
    print(f"Total structured columns generated: {len(headers)}")
    print("First 10 columns:", headers[:10])

    # Diagnostic Phase

    file_path = f"{folder_path}subject{subject_num}.dat"

    print("Reading from "+file_path)

    print("Loading raw subject file (this might take a few seconds because it is a massive telemetry file)...")
    # We use sep='\s+' because the text file uses variable numbers of spaces to separate numbers
    df_raw = pd.read_csv(file_path, sep=r'\s+', header=None)
    df_raw.columns = headers

    print(f"\n--- RAW DATA DIAGNOSTICS FOR SUBJECT "+subject_num+" ---")
    print(f"Total raw rows: {len(df_raw):,}")
    print(f"Total continuous tracking time: {df_raw['timestamp'].max() / 60:.2f} minutes")

    # 1. Check for missing data (NaNs) by percentage
    missing_pct = df_raw.isnull().mean() * 100
    print("\nTop 5 Columns with the Highest Percentage of Missing Data (NaNs):")
    print(missing_pct.sort_values(ascending=False).head(5))

    # # 2. Check the distribution of activities this person did
    # print("\nRaw row counts per Activity ID:")
    # activity_counts = df_raw['activity_id'].value_counts()
    # for act_id, count in activity_counts.items():
    #     name = activity_map.get(act_id, 'Unknown Activity')
    #     print(f"  ID {act_id:<2} ({name}): {count:,} rows")

    return df_raw

def extract_intervals(subject_num:int, df_intervals:pd.DataFrame, folder_path:str='../PAMAP2_Dataset/Protocol/')->pd.DataFrame:
    
    df_raw = load_subject(subject_num, folder_path)

    #find where activity ID changes to divide intervals
    df_raw['activity_change'] = df_raw['activity_id'].diff()
    (df_raw['activity_change'] != 0).index
    a = df_raw.loc[df_raw['activity_change'] != 0.0].index.tolist()
    a = a[1:]

    #set start time and activity ID for first interval
    n1 = 0
    t1 = df_raw['timestamp'][n1]
    activity = df_raw['activity_id'][n1]

    #loop through interval transitions
    for i in a:
        #set end time for interval
        n2 = i-1
        t2 = df_raw['timestamp'][n2]
        #write interval to dataframe
        df_add = pd.DataFrame.from_records([{'activity_id':activity,
                                            'subject_id':subject_num,
                                            'length':t2-t1,
                                            't1':t1,
                                            't2':t2,
                                            'n1':n1,
                                            'n2':n2}])

        df_intervals = pd.concat([df_intervals, df_add], ignore_index=True)
        
        #set start time and activity ID for next interval
        n1 = i
        t1 = df_raw['timestamp'][i]
        activity = df_raw['activity_id'][i]

    #close and write final interval
    t2 = df_raw.tail(1)['timestamp'].values[0]
    df_add = pd.DataFrame.from_records([{'activity_id':activity,
                                        'subject_id':subject_num,
                                        'length':t2-t1,
                                        't1':t1,
                                        't2':t2,
                                        'n1':n1,
                                        'n2':n2}])
    df_intervals = pd.concat([df_intervals, df_add], ignore_index=True)
    
    return df_intervals

def interval_stats(df_intervals,activity_id):
    #function to print simple statistics of activity intervals
    'df_intervals: dataframe of activity intervals, as created by extract_intervals (pandas.DataFrame)'
    'activity_id: numerical ID of activity to print stats for (int)'

    activity_map = get_activityMap()
    activity = activity_map[activity_id]
    print('Interval statistics for '+activity+':')

    df_select = df_intervals[df_intervals['activity_id']==activity_id]
    lengths = df_select['length'].values

    num = len(lengths)
    print('Number of intervals: '+str(num))
    if num != 0:
        maxTime = np.max(lengths)
        minTime = np.min(lengths)
        meanTime = np.mean(lengths)
        print('Max time (s): '+str(maxTime))
        print('Min time (s): '+str(minTime))
        print('Mean time (s): '+str(meanTime))
    else:
        maxTime = None
        minTime = None
        meanTime = None
    
    return (num,maxTime,minTime,meanTime)


def interp_data(df_inpt:pd.DataFrame,t_window=3,columns=None):

    # df_inpt: input dataframe in which we want to. pass in the original dataframe without any segmenting, else the interpolation bounds could be incorrect
    # t_window: consecutive NaNs over intervals smaller than this window (in seconds) will be interpolated
    # columns: list of columns within which to interpolate. default inteprolates in all columns, except heart rate


    df = df_inpt.copy() #work on a copy dataframe

    if columns == None:
        columns = list(df.columns.values)
    else:
        columns = columns.copy()

    # we will want to deal with heart rate sampling as a special case, since it is sampled at a low frequency
    # skip for now, implement here or elsewhere when we decide how to proceed

    try:
        columns.remove('timestamp')
    except ValueError:
        pass
    try:
        columns.remove('heart_rate')
    except ValueError:
        pass
    try:
        columns.remove('activity_id')
    except ValueError:
        pass
    for column in columns:

 
        # find which rows are NaN
        null_search = df[column].isnull()
        null_list = null_search[null_search].index.values

        missing_pct = df[column].isnull().mean() * 100
        #I know there is cleaner string formatting but I forget the syntax and I'm not looking it up now
        print('Column '+column+' has '+str(round(missing_pct,2))+'% NaNs.')
        print('Interpolating in '+column+'...')
        if len(null_list) != 0:
            i = null_list[0]
            # group NaN rows by consecutive sequences
            for k, g in groupby(enumerate(null_list), lambda x: x[0]-x[1]):
                n_set = list(map(itemgetter(1), g))
                # skip NaNs at the beginning or end of sequence
                if n_set[0] == df.index.values[0] or n_set[-1] == df.index.values[-1]:
                    pass
                elif df['timestamp'][n_set[-1]] - df['timestamp'][n_set[0]] < t_window:
                    #get values for points before and after missing data
                    n1 = n_set[0]-1
                    n2 = n_set[-1]+1
                    t1 = df['timestamp'][n1]
                    t2 = df['timestamp'][n2]
                    y1 = df[column][n1]
                    y2 = df[column][n2]
                    #slope for interpolation
                    m = (y2-y1)/(t2-t1)

                    # calculate and insert inteprolated values
                    tvals = np.array(df.loc[n_set]['timestamp'].values)
                    df.loc[n_set,column] = y1 + m*(tvals - t1)
        
        missing_pct = df[column].isnull().mean() * 100
        print('Column '+column+' now has '+str(round(missing_pct,2))+'% NaNs!\n')

    return df


def interp_data_new(df_inpt:pd.DataFrame,columns=None,t_window=3):

    # df_inpt: input dataframe in which we want to. pass in the original dataframe without any segmenting, else the interpolation bounds could be incorrect
    # t_window: consecutive NaNs over intervals smaller than this window (in seconds) will be interpolated
    # columns: list of columns within which to interpolate. default inteprolates in all columns, except heart rate


    df = df_inpt.copy() #work on a copy dataframe

    df.set_index('timestamp')

    if columns == None:
        columns = list(df.columns.values)
    else:
        columns = columns.copy()

    # we will want to deal with heart rate sampling as a special case, since it is sampled at a low frequency
    # skip for now, implement here or elsewhere when we decide how to proceed

    try:
        columns.remove('timestamp')
    except ValueError:
        pass
    try:
        columns.remove('heart_rate')
    except ValueError:
        pass
    try:
        columns.remove('activity_id')
    except ValueError:
        pass
    for column in columns:

 
        # find which rows are NaN
        null_search = df[column].isnull()
        null_list = null_search[null_search].index.values

        missing_pct = df[column].isnull().mean() * 100
        #I know there is cleaner string formatting but I forget the syntax and I'm not looking it up now
        print('Column '+str(column)+' has '+str(round(missing_pct,2))+'% NaNs.')
        print('Interpolating in '+str(column)+'...')

        df[column] = df[column].interpolate(method='index')
        
        missing_pct = df[column].isnull().mean() * 100
        print('Column '+str(column)+' now has '+str(round(missing_pct,2))+'% NaNs!\n')

    return df