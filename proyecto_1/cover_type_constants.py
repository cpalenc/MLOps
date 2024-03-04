 
NUMERIC_FEATURE_KEYS = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Vertical_Distance_To_Hydrology']
 
FEATURE_RANGE_0_255 = ['Hillshade_9am', 'Hillshade_Noon']
 
FEATURE_RANGE_0_90 = ['Slope']
 
LABEL_KEY= 'Cover_Type'
 
def transformed_name(key):
    return key + '_xf'
