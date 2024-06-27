import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers
import os  
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np

LABEL_KEY ='quality'

def _transformed_name(key):
    return key.replace(' ', '_').lower() + '_xf'

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

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

def model_builder():
    """Build machine learning model"""
    input_layers = [
        tf.keras.Input(shape=(1,), name=_transformed_name(f), dtype=tf.float32)
        for f in [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
    ]
    
    concatenated_features = tf.keras.layers.concatenate(input_layers)
    x = layers.Dense(128, activation='relu')(concatenated_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(1)(x)
    
    # Define the model
    model = tf.keras.Model(inputs=input_layers, outputs=outputs)
    
    model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=[tf.keras.losses.MeanSquaredError()]
)
    
    # print(model)
    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # Get predictions using the transformed features
        return model(transformed_features)
    
        # Sanitize input names for SavedModel signature
        sanitized_input_names = [f.replace(' ', '_').lower() + '_xf' for f in feature_spec.keys()]
    
        serving_default_signature = serve_tf_examples_fn.get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
        serving_default_signature._input_signature[0].name = 'examples'  # Rename input tensor
    
        # Assign sanitized names to input tensors
        for i, input_tensor in enumerate(serving_default_signature.inputs):
            input_tensor._name = sanitized_input_names[i]  # Assign sanitized name to input tensor

    return serve_tf_examples_fn




def run_fn(fn_args: FnArgs) -> None:
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_loss', mode='max', verbose=1, save_best_only=True)
    
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    model = model_builder()

    model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            steps_per_epoch = 1000, 
            validation_steps= 1000,
            epochs=10)
            
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }
    
    # Save the model with signatures
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
