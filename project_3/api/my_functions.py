# 1. Library imports
import numpy as np

# 2. Values to standardize variables.
scaler_min_max = {'min': np.array([5., 1., 1., 1., 0., 0., 0., 0., 0., 1.]), 
                  'max': np.array([ 95.,  28.,  14., 132.,   6.,  21.,   9.,   9.,   9.,  16.])}
scaler_std = {'mean': np.array([0.67732942, 0.10153072, 0.26144829, 0.32117825, 0.22342019, 
                                0.03062968, 0.48420805, 0.41864764, 0.38523188, 0.42931955]), 
              'std': np.array([0.17707144, 0.19665707, 0.22977266, 0.15034933, 0.28392833,
                               0.0605452 , 0.23028769, 0.20487616, 0.21306782, 0.12835119])}

# 3. Functions to standardize and predict.

def func_transform(user_input):
    user_input_scaled = (user_input - scaler_min_max['min']) / (scaler_min_max['max'] - scaler_min_max['min'])
    user_input_scaled = (user_input_scaled - scaler_std['mean']) / scaler_std['std']
    return user_input_scaled
