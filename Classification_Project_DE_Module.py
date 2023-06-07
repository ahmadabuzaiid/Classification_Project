import numpy as np
import pandas as pd
from datetime import datetime
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def cca_imp_fn(df, feat):
    df.dropna(subset=feat,inplace=True)
    return df

def drop_col(df, feat):
    df.drop([feat], axis=1, inplace=True)
    return df

def missing_imp_fn(df, feat):
    if (df[feat].dtypes == object):
        df[feat].fillna(value='missing',inplace=True)
    else:
        df[feat].fillna(value=-1,inplace=True)
    return df

def most_freq_imp_fn(df, feat):
    freq = df[feat].value_counts().index[0]
    df[feat].fillna(value=freq,inplace=True)
    return df

def dateTime_fn(day):
    date_arr = day.split('/')
    return datetime(int(date_arr[2]), int(date_arr[1]), int(date_arr[0]))

def weekNum_fn(day):
    return int(day.strftime("%V"))

def rare_label_encoder(df, rare_feat):
    for feat in rare_feat:
        _list = df[feat].value_counts().iloc[::-1]
        count_sum = 0
        rare_list = []
        
        for i in range(len(_list)):
            count_sum = count_sum + _list[i]
            if (count_sum/len(df.index) <= 0.05):
                rare_list.append(_list.index[i])
            i = i + 1
            
        df[feat].replace(set(rare_list),'Rare',inplace=True)
    return df

def bin_encoder(df, bin_list):
    bin_encoder = BinaryEncoder(cols=bin_list)
    newdata = bin_encoder.fit_transform(df[bin_list])
    df_bin_encoded = pd.concat([df, newdata], axis = 1)
    df_bin_encoded = df_bin_encoded.drop(bin_list, axis = 1)
    return df_bin_encoded

def one_hot_encoder(df, hot_feat):
    encoded_colm = pd.get_dummies(df, columns = hot_feat)
    df_hot_encoded = pd.concat([df, encoded_colm], axis=1)
    df_hot_encoded = df_hot_encoded.drop(hot_feat, axis=1)
    return df_hot_encoded

def label_encoder_fn(df, label_feat):
    label_encoder = LabelEncoder()
    df[label_feat] = label_encoder.fit_transform(df[label_feat])
    return df

def scal_fn(df, scal_feat):
    for feat in scal_feat:
        df[feat] = MinMaxScaler().fit_transform(df[[feat]]) 
    return df

def standardize_fn(df, stand_feat):
    for feat in stand_feat:
        df[feat] = StandardScaler().fit_transform(df[[feat]])
    return df

def weekEnd_fn(day):
    if (day == 'Saturday' or day == 'Sunday'):
        return 1
    else:
        return 0

def severity_encoder(severity):
    if (severity == 'Slight'):
        return 0
    elif (severity == 'Serious'):
        return 1
    else:
        return 2
    
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    total_minutes = hours * 60 + minutes
    return total_minutes

def discretize_time_in_minutes(minutes):
    if minutes < 360:  # Morning (6:00 - 11:59)
        return "Morning"
    elif minutes < 720:  # Afternoon (12:00 - 17:59)
        return "Afternoon"
    elif minutes < 1080:  # Evening (18:00 - 23:59)
        return "Evening"
    else:  # Night (0:00 - 5:59)
        return "Night"

def DE_fn():

    df = pd.read_csv('./2018_Accidents_UK.csv',low_memory=False)

    df.drop_duplicates(subset=None, keep='first', inplace=True)
    df = df.drop(df[df['second_road_class'] == '9'].index)
    df.replace({-1,'-1','Data missing or out of range','Unallocated'},np.nan,inplace=True)
    df.replace('first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ','0',inplace=True)

    df['first_road_number'] = pd.to_numeric(df['first_road_number'])
    df['second_road_number'] = pd.to_numeric(df['second_road_number'])
    df['speed_limit'] = pd.to_numeric(df['speed_limit'])

    for feat in df.columns:
        if (df[feat].isnull().sum()/len(df.index)) <= 0.01:
            df = cca_imp_fn(df, feat)
        elif (df[feat].isnull().sum()/len(df.index)) >= 0.8:
            df = drop_col(df, feat)
        elif (df[feat].isnull().sum()/len(df.index)) > 0.05:
            df = missing_imp_fn(df, feat)
        else:
            df = most_freq_imp_fn(df, feat)

    clf = LocalOutlierFactor()
    x = df[['number_of_vehicles','number_of_casualties']].values
    y_pred = clf.fit_predict(x)

    out_mask = [True if l == -1 else False for l in y_pred]

    for i in range(len(x[out_mask, 0])):
        df.drop(df.loc[(df['number_of_vehicles'] == x[out_mask, 0][i]) & (df['number_of_casualties'] == x[out_mask, 1][i])].index, inplace=True)

    df['dateTime'] = df['date'].map(lambda day: dateTime_fn(day))
    df['week_end'] = df['day_of_week'].map(lambda day: weekEnd_fn(day))
    df['week_number'] = df['dateTime'].map(lambda day: weekNum_fn(day))
    df['time_in_mins'] = df['time'].map(lambda time: time_to_minutes(time))
    df['time_discrete'] = df['time_in_mins'].map(lambda time: discretize_time_in_minutes(time))
    df['accident_severity'] = df['accident_severity'].map(lambda severity: severity_encoder(severity))

    df = label_encoder_fn(df, 'speed_limit')
    
    rare_feat = ['police_force', 'local_authority_district', 'local_authority_ons_district', 'local_authority_highway', 'junction_control', 'pedestrian_crossing_human_control', 'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site', 'carriageway_hazards']
    df = rare_label_encoder(df, rare_feat)

    bin_feat = ['police_force','day_of_week','local_authority_district','local_authority_ons_district','local_authority_highway','first_road_class','road_type','junction_detail','junction_control','pedestrian_crossing_human_control','pedestrian_crossing_physical_facilities','light_conditions','weather_conditions','road_surface_conditions','did_police_officer_attend_scene_of_accident','trunk_road_flag','week_number','time_discrete']
    df = bin_encoder(df, bin_feat)

    hot_feat = ['special_conditions_at_site','carriageway_hazards','urban_or_rural_area']
    df = one_hot_encoder(df, hot_feat)
    df = df.loc[:, ~df.columns.duplicated()]

    stand_feat = ['number_of_vehicles', 'number_of_casualties','time_in_mins']
    df = standardize_fn(df, stand_feat)

    df.to_csv('2018_Accidents_UK_Final.csv', index=False)