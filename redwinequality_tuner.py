from typing import NamedTuple, Dict, Any, Text
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from keras_tuner.engine import base_tuner
from kerastuner.tuners import Hyperband
from keras_tuner.engine.base_tuner import BaseTuner
import tensorflow_transform as tft
import tensorflow as tf

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', BaseTuner), ('fit_kwargs', Dict[Text, Any])])

LABEL_KEY='quality'

def _transformed_name(key):
    return key.replace(' ', '_').lower() + '_xf'
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Define input function to read CSV files
def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs=None,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = _transformed_name(LABEL_KEY))
    return dataset.repeat()

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.

    Args:
        fn_args: Holds args used to tune models as name/value pairs.

    Returns:
        A TunerFnResult that contains the tuner and fit_kwargs.
    """
    # Load the transformed data
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output)

    # Define the model builder function
    def model_builder(hp):
        """Build machine learning model with tunable hyperparameters"""
        # Define hyperparameters
        hp_units = hp.Int('units', min_value=128, max_value=512, step=32)
        hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])

        input_layers = [
            tf.keras.Input(shape=(1,), name=_transformed_name(f), dtype=tf.float32)
            for f in [
                'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol'
            ]
        ]
        
        concatenated_features = tf.keras.layers.concatenate(input_layers)
        x = concatenated_features
        x = layers.Dense(hp_units, activation='relu')(concatenated_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(hp_units//2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(hp_units//2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        # Output layer with softmax activation for multi-class classification
        outputs = layers.Dense(1)(x)
        
        # Define the model
        model = tf.keras.Model(inputs=input_layers, outputs=outputs)
        
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(hp_learning_rate),
            metrics=[tf.keras.losses.MeanSquaredError()]
        )
        
        return model

    # Define the hyperband tuner
    tuner = Hyperband(model_builder,
                         objective='val_mean_squared_error',
                         max_epochs=10,
                         factor=3,
                         directory=fn_args.working_dir,
                         project_name='kt_hyperband')

    # Set fit arguments for the tuner
    early_stopping = EarlyStopping(monitor='val_mean_squared_error',  mode='max', min_delta=0.001, patience=5, verbose=1)

    fit_kwargs = {
        "callbacks": [early_stopping],
        'x': train_set,
        'validation_data': val_set,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
