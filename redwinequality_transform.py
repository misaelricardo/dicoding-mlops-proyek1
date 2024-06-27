import tensorflow as tf
import tensorflow_transform as tft
def _transformed_name(key):
    return key.replace(' ', '_').lower() + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input features."""
    outputs = {}

    # Normalize numerical features
    for feature_name in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                         'pH', 'sulphates', 'alcohol']:
        outputs[_transformed_name(feature_name)] = inputs[feature_name]

    # Leave 'quality' as it is (treat it as a categorical feature)
    outputs[_transformed_name('quality')] = inputs['quality']

    return outputs
