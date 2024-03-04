import tensorflow as tf
import tensorflow_transform as tft
import cover_type_constants

# Unpack the contents of the constants module

_NUMERIC_FEATURE_KEYS = cover_type_constants.NUMERIC_FEATURE_KEYS

_FEATURE_RANGE_0_255 = cover_type_constants.FEATURE_RANGE_0_255

_FEATURE_RANGE_0_90 = cover_type_constants.FEATURE_RANGE_0_90

_LABEL_KEY = cover_type_constants.LABEL_KEY

_transformed_name = cover_type_constants.transformed_name 
# Define the transformations
def preprocessing_fn(inputs):
    outputs = {}
 
    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(
            inputs[key])
   
    # Bucketize these features
    for key in _FEATURE_RANGE_0_255:
        outputs[_transformed_name(key)] = tft.scale_by_min_max(
            inputs[key], 0.0, 255.0)
 
    # Convert strings to indices in a vocabulary
    for key in _FEATURE_RANGE_0_90:
        outputs[_transformed_name(key)] = tft.scale_by_min_max(
            inputs[key], 0.0, 90.0)
 
    # Convert the label strings to an index
    outputs[_transformed_name(_LABEL_KEY)] = (inputs[_LABEL_KEY])
 
    return outputs
