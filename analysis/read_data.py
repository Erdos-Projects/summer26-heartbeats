import pandas as pd

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

def load_subject(subject_num):

    headers = get_pamap2_headers()
    print(f"Total structured columns generated: {len(headers)}")
    print("First 10 columns:", headers[:10])

    # Diagnostic Phase

    file_path = '../data/PAMAP2_Dataset/Protocol/subject'+subject_num+'.dat'

    print("Reading from "+file_path)

    print("Loading raw subject file (this might take a few seconds because it is a massive telemetry file)...")
    # We use sep='\s+' because the text file uses variable numbers of spaces to separate numbers
    df_raw = pd.read_csv(file_path, sep='\s+', header=None)
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

def extract_intervals(subject_num,df_intervals,verbose=False):
    
    df_raw = load_subject(subject_num)

    #find where activity ID changes to divide intervals
    df_raw['activity_change'] = df_raw['activity_id'].diff()
    (df_raw['activity_change'] != 0).index
    a = df_raw.loc[df_raw['activity_change'] != 0.0].index.tolist()
    a = a[1:]

    #set start time and activity ID for first interval
    act_ind = 0
    t1 = df_raw['timestamp'][0]
    activity = df_raw['activity_id'][0]

    #loop through interval transitions
    for i in a:
        #set end time for interval
        t2 = df_raw['timestamp'][i-1]
        #write interval to dataframe
        df_add = pd.DataFrame.from_records([{'activity_id':activity,
                                                                            'subject_id':subject_num,
                                                                            'length':t2-t1,
                                                                            't1':t1,
                                                                            't2':t2}])

        df_intervals = pd.concat([df_intervals, df_add], ignore_index=True)
        
        #set start time and activity ID for next interval
        t1 = df_raw['timestamp'][i]
        activity = df_raw['activity_id'][i]

    #close and write final interval
    t2 = df_raw.tail(1)['timestamp'].values[0]
    df_add = pd.DataFrame.from_records([{'activity_id':activity,
                                                                        'subject_id':subject_num,
                                                                        'length':t2-t1,
                                                                        't1':t1,
                                                                        't2':t2}])
    df_intervals = pd.concat([df_intervals, df_add], ignore_index=True)
    
    return df_intervals

def intervalStats_activity(df_intervals,activity_id):
    #in progress

    activity_map = get_activityMap()
    activity = activity_map[activity_id]
    print('Interval statistics for '+activity+':')

    df_select = df_intervals[df_intervals['activity_id']==activity_id]




def intervalStats_subject():
    #in progress
    pass