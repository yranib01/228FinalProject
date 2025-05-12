import scipy.io
import datetime
import pandas

f_samp = 1500

def idxtot(idx, f_samp=1500, start=datetime.datetime(1996, 5, 10, hour=23, minute=15, second=0)):
    delta = datetime.timedelta(milliseconds=1000 * idx/(f_samp))

    return start + delta

def dftodict(df):
    mins_to_range = {}
    dt_to_range = {}
    jdaytoday = lambda x: datetime.date(1996, 5, 10) if x == 131 else datetime.date(1996, 5, 11)
    for i in df.index:
        range = df.at[i, "Range(km)"]
        mins = df.at[i, "Duration"]
        jday = df.at[i, "Jday"]
        day = jdaytoday(jday)
        time = datetime.datetime.strptime(df.at[i, "Time"], "%H:%M").time()

        dt = datetime.datetime.combine(day, time)

        mins_to_range[mins] = range
        dt_to_range[dt] = range

    return mins_to_range, dt_to_range

ranges = pandas.read_csv('SproulToVLA.S5.txt', sep='\s+')

s5 = scipy.io.loadmat("s5.mat")

s5 = s5['s5']

ranges_s5 = dftodict(ranges)

