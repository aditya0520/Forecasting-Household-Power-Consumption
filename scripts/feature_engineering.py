import pandas as pd
import numpy as np

def create_features(df):


    df['DateTime'] = df.index
    
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Year'] = df['DateTime'].dt.year

    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    data = df[['Global_active_power', 'DayOfWeek_sin', 'DayOfWeek_cos']].copy()
    return data
