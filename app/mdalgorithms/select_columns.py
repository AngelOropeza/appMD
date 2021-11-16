import numpy as np

def select_columns(df, selected_cols):
    if len(selected_cols) == 0:
        return np.array(df)
    return np.array(df[selected_cols])