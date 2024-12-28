import pandas as pd

def load_and_clean_data(file_name):

    df = pd.read_csv(
        file_name,
        sep=';',  
        parse_dates={'DateTime': ['Date', 'Time']},
        infer_datetime_format=True,  
        low_memory=False 
    )


    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

    df = df.dropna()

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
    df = df.reindex(full_index)

    df['Global_active_power'] = df['Global_active_power'].interpolate(method='time')

    return df
