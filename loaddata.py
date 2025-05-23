import pandas as pd
import scipy.io
import datetime
import numpy as np

F_SAMP = 1500 # sampling frequency : constant -> Letter
START_DT = datetime.datetime(1996, 5, 10, hour=23, minute=15, second=0)


# 1) index -> datetime
def idx_to_time(idx : int,
                f_samp : int = F_SAMP,
                start : datetime.datetime = START_DT) -> datetime.datetime:
    # Sample index -> datetime
    delta = datetime.timedelta(milliseconds=1000 * idx/(f_samp))

    return start + delta

# 2) DataFrame -> dictionary -> convert two dictionaries 
def df_to_range_dicts(df : pd.DataFrame):
    """
    SproulToVAL.txt DataFrame ->
    - mins_to_range : {Duration(min) -> Range(km)}
    - dt_to_range : {datetime -> Range(km)}
    """
    mins_to_range , dt_to_range = {}, {}
    
    for _, row in df.iterrows():
        rng  = row["Range(km)"]
        mins = row["Duration"]
        # Jday -> datetime.date
        day   = datetime.date(1996, 5, 10) if row["Jday"] == 131 else datetime.date(1996, 5, 11)
        time  = datetime.datetime.strptime(row["Time"], "%H:%M").time()
        dt    = datetime.datetime.combine(day, time)
        mins_to_range[mins] = rng
        dt_to_range[dt]     = rng
    return mins_to_range, dt_to_range

# 3) Load txt, mat file
range_df  =  pd.read_csv('SproulToVLA.S5.txt', sep=r"\s+")
mins2rng, dt2rng = df_to_range_dicts(range_df)

mat    = scipy.io.loadmat("s5.mat")
s5     = mat['s5'] # s5 is a 2D array : s5.shape = (6,750,000, 21) , len(s5) = 6,750,000

# Make the shape information as vairables
N_TIMESTEPS, N_SENSORS = s5.shape      # N_TIMESTEPS = 6,750,000, N_SENSORS = 21