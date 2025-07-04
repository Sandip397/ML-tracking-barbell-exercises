import pandas as pd
from glob import glob

#-----------------------------
#Read CSV file 

# single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

# single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# #---------------------------------
# #List all data in data/raw/MetaMotion
# files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)

# #----------------------------------
# #Extract Feature from file name 

# data_path = "../../data/raw/MetaMotion/"
# f = files[0]

# # # Extract filename from the full path using both possible path separators
# # filename = f.split('/')[-1].split('\\')[-1]

# # Extract participant information (first character before '-')
# participant = f.split('/')[-1].split('\\')[-1].split("-")[0]
# label = f.split("-")[1]
# category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

# df = pd.read_csv(f)

# df["participant"] = participant
# df["label"] = label
# df["category"] = category

# #-----------------------------------
# #Read all the file 

# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()

# acc_set = 1
# gyr_set = 1

# for f in files:
#     participant = f.split('/')[-1].split('\\')[-1].split("-")[0]
#     label = f.split("-")[1]
#     category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
  
#     df = pd.read_csv(f)
    
#     df["participant"] = participant
#     df["label"] = label
#     df["category"] = category
    
#     if "Accelerometer" in f:
#         df["set"] = acc_set
#         acc_set += 1
#         acc_df = pd.concat([acc_df, df])
        
#     if "Gyroscope" in f:
#         df["set"] = gyr_set
#         gyr_set += 1
#         gyr_df = pd.concat([gyr_df, df])
        
# #-----------------------------------
# #Working with datetime

# acc_df.info()

# pd.to_datetime(df["epoch (ms)"], unit="ms")

# acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df["elapsed (s)"]

# del gyr_df["epoch(ms)"]
# del gyr_df["time (01:00)"]
# del gyr_df["elapsed (s)"]

#-----------------------------------------------
# Turn into function 

files = glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_files(files): 
    
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split('/')[-1].split('\\')[-1].split("-")[0]
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
            
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
        
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df,gyr_df = read_data_from_files(files)

#-----------------------------------------------
# Merging datasets

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

#Rename columns

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",    
]

#----------------------------------------------------
#Resample data[frequency converation]
#Data Sample was taken on the basis of frequency

#Accelatrometer :   12.500Hz(1/12.5 It was measuring data evry 0.08 second )
#gyroscape:         25.00Hz (1/25 It was measuring every 0.04 second)
sampling  = {
    "acc_x" : "mean",
    "acc_y" : "mean",
    "acc_z" : "mean",
    "gyr_x" : "mean",
    "gyr_y" : "mean",
    "gyr_z" : "mean",
    "participant" : "last",
    "label" : "last",
    "category" : "last",
    "set" : "last",        
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

#Split by day 
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()
#

#-------------------------------------
# Export datset 

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
                
    

