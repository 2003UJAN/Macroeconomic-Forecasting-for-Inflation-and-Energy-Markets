"""
Model Architectures for Macroeconomic Forecasting
==================================================

This module contains deep learning model architectures for time series forecasting:
1. Vanilla LSTM (Baseline)
2. Bidirectional LSTM with Attention Mechanism
3. CNN-LSTM Hybrid
4. Transformer-based Model

Author: Your Name
Date: October 2025
Project: Deep Learning for Inflation and Oil Price Forecasting
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for LSTM models.
    
    This layer implements a simple attention mechanism that learns to weight
    the importance of different time steps in the sequence.
    
    Args:
        **kwargs: Additional keyword arguments for the Layer class
    
    Input shape:
        3D tensor with shape: (batch_size, time_steps, features)
    
    Output shape:
        2D tensor with shape: (batch_size, features)
    
    Example:
        >>> attention = AttentionLayer()
        >>> output = attention(lstm_output)
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        """
        Build the layer by creating weight matrices.
        
        Args:
            input_shape: Shape of input tensor
        """
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        """
        Forward pass of the attention layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted sum of input sequence
        """
        # Calculate attention scores
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        # Apply softmax to get attention weights
        a = tf.nn.softmax(e, axis=1)
        # Apply attention weights
        output = x * a
        # Sum along time dimension
        return tf.reduce_sum(output, axis=1)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(AttentionLayer, self).get_config()
        return config


# ============================================================================
# MODEL 1: VANILLA LSTM (BASELINE)
# ============================================================================

def create_vanilla_lstm(n_steps=12, n_features=6, 
                        lstm_units=[64, 32], 
                        dense_units=[16],
                        dropout_rate=0.2,
                        learning_rate=0.001):
    """
    Create a simple vanilla LSTM model for time series forecasting.
    
    This model serves as a baseline for comparison with more complex architectures.
    It uses stacked LSTM layers with dropout for regularization.
    
    Args:
        n_steps (int): Number of time steps in input sequence (lookback period)
        n_features (int): Number of features in each time step
        lstm_units (list): List of units for each LSTM layer
        dense_units (list): List of units for dense layers
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled LSTM model
    
    Model Architecture:
        Input -> LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Output(1)
    
    Example:
        >>> model = create_vanilla_lstm(n_steps=12, n_features=6)
        >>> model.summary()
    """
    
    model = models.Sequential(name='Vanilla_LSTM')
    
    # First LSTM layer (return sequences for stacking)
    model.add(layers.LSTM(
        lstm_units[0], 
        activation='relu',
        return_sequences=True,
        input_shape=(n_steps, n_features),
        name='lstm_1'
    ))
    model.add(layers.Dropout(dropout_rate, name='dropout_1'))
    
    # Second LSTM layer
    model.add(layers.LSTM(
        lstm_units[1],
        activation='relu',
        name='lstm_2'
    ))
    model.add(layers.Dropout(dropout_rate, name='dropout_2'))
    
    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(1, name='output'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


# ============================================================================
# MODEL 2: BIDIRECTIONAL LSTM WITH ATTENTION
# ============================================================================

def create_bilstm_attention(n_steps=12, n_features=6,
                           lstm_units=[128, 64],
                           dense_units=[32],
                           dropout_rate=0.3,
                           learning_rate=0.001):
    """
    Create Bidirectional LSTM with Attention Mechanism.
    
    This model processes sequences in both forward and backward directions,
    capturing temporal dependencies from both past and future contexts.
    The attention mechanism learns to focus on the most relevant time steps.
    
    Args:
        n_steps (int): Number of time steps in input sequence
        n_features (int): Number of features in each time step
        lstm_units (list): List of units for each BiLSTM layer
        dense_units (list): List of units for dense layers
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled BiLSTM-Attention model
    
    Model Architecture:
        Input -> BiLSTM(128) -> Dropout -> BiLSTM(64) -> Dropout -> 
        Attention -> Dense(32) -> Dropout -> Output(1)
    
    Example:
        >>> model = create_bilstm_attention()
        >>> model.fit(X_train, y_train, epochs=100)
    """
    
    # Input layer
    inputs = layers.Input(shape=(n_steps, n_features), name='input')
    
    # First Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True),
        name='bidirectional_lstm_1'
    )(inputs)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Second Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[1], return_sequences=True),
        name='bidirectional_lstm_2'
    )(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Attention mechanism
    x = AttentionLayer(name='attention')(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate * 0.67, name=f'dropout_{i+3}')(x)
    
    # Output layer
    outputs = layers.Dense(1, name='output')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


# ============================================================================
# MODEL 3: CNN-LSTM HYBRID
# ============================================================================

def create_cnn_lstm(n_steps=12, n_features=6,
                   conv_filters=[64, 128],
                   kernel_size=3,
                   pool_size=2,
                   lstm_units=[100, 50],
                   dense_units=[25],
                   dropout_rate=0.3,
                   learning_rate=0.001):
    """
    Create CNN-LSTM Hybrid Model.
    
    This model combines Convolutional Neural Networks for feature extraction
    with LSTM for temporal sequence modeling. CNN layers extract local patterns
    while LSTM captures long-term dependencies.
    
    Args:
        n_steps (int): Number of time steps in input sequence
        n_features (int): Number of features in each time step
        conv_filters (list): Number of filters for each Conv1D layer
        kernel_size (int): Size of convolutional kernel
        pool_size (int): Size of max pooling window
        lstm_units (list): List of units for each LSTM layer
        dense_units (list): List of units for dense layers
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled CNN-LSTM model
    
    Model Architecture:
        Input -> Conv1D(64) -> MaxPool -> Conv1D(128) -> MaxPool ->
        LSTM(100) -> Dropout -> LSTM(50) -> Dropout -> Dense(25) -> Output(1)
    
    Example:
        >>> model = create_cnn_lstm(n_steps=12, n_features=6)
        >>> history = model.fit(X_train, y_train, validation_data=(X_test, y_test))
    """
    
    # Input layer
    inputs = layers.Input(shape=(n_steps, n_features), name='input')
    
    # First CNN block
    x = layers.Conv1D(
        filters=conv_filters[0],
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
        name='conv1d_1'
    )(inputs)
    x = layers.MaxPooling1D(pool_size=pool_size, name='maxpool_1')(x)
    
    # Second CNN block
    x = layers.Conv1D(
        filters=conv_filters[1],
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
        name='conv1d_2'
    )(x)
    x = layers.MaxPooling1D(pool_size=pool_size, name='maxpool_2')(x)
    
    # First LSTM layer
    x = layers.LSTM(lstm_units[0], return_sequences=True, name='lstm_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Second LSTM layer
    x = layers.LSTM(lstm_units[1], name='lstm_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(1, name='output')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


# ============================================================================
# MODEL 4: TRANSFORMER MODEL
# ============================================================================

def create_transformer_model(n_steps=12, n_features=6,
                            head_size=256,
                            num_heads=4,
                            ff_dim=4,
                            num_transformer_blocks=2,
                            mlp_units=[128],
                            dropout=0.25,
                            learning_rate=0.001):
    """
    Create Transformer-based Model for Time Series Forecasting.
    
    This model uses multi-head self-attention mechanisms to capture complex
    temporal dependencies without recurrence. Transformers excel at learning
    long-range dependencies and have shown superior performance in recent
    time series forecasting tasks.
    
    Args:
        n_steps (int): Number of time steps in input sequence
        n_features (int): Number of features in each time step
        head_size (int): Size of each attention head
        num_heads (int): Number of attention heads
        ff_dim (int): Hidden layer size in feed forward network
        num_transformer_blocks (int): Number of transformer blocks
        mlp_units (list): List of units for MLP head
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled Transformer model
    
    Model Architecture:
        Input -> [Transformer Block x2] -> GlobalAvgPool -> 
        Dense(128) -> Dropout -> Output(1)
        
        Each Transformer Block:
            MultiHeadAttention -> Add&Norm -> FeedForward -> Add&Norm
    
    Example:
        >>> model = create_transformer_model(n_steps=12, n_features=6)
        >>> model.fit(X_train, y_train, epochs=100, batch_size=32)
    
    References:
        - Vaswani et al. (2017): "Attention is All You Need"
        - Recent applications show 20-30% improvement over LSTM for economic forecasting
    """
    
    # Input layer
    inputs = layers.Input(shape=(n_steps, n_features), name='input')
    x = inputs
    
    # Transformer blocks
    for i in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout,
            name=f'multi_head_attention_{i+1}'
        )(x, x)
        
        # Add & Normalize
        x1 = layers.Add(name=f'add_1_{i+1}')([x, attention_output])
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f'layer_norm_1_{i+1}')(x1)
        
        # Feed Forward Network
        ffn = layers.Dense(ff_dim, activation="relu", name=f'ffn_dense_1_{i+1}')(x1)
        ffn = layers.Dropout(dropout, name=f'ffn_dropout_{i+1}')(ffn)
        ffn = layers.Dense(n_features, name=f'ffn_dense_2_{i+1}')(ffn)
        
        # Add & Normalize
        x = layers.Add(name=f'add_2_{i+1}')([x1, ffn])
        x = layers.LayerNormalization(epsilon=1e-6, name=f'layer_norm_2_{i+1}')(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # MLP head
    for i, units in enumerate(mlp_units):
        x = layers.Dense(units, activation="relu", name=f'mlp_dense_{i+1}')(x)
        x = layers.Dropout(dropout, name=f'mlp_dropout_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(1, name='output')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name='Transformer')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


# ============================================================================
# MODEL 5: ENCODER-DECODER LSTM
# ============================================================================

def create_encoder_decoder_lstm(n_steps=12, n_features=6,
                               encoder_units=[128, 64],
                               decoder_units=64,
                               dense_units=[32],
                               dropout_rate=0.3,
                               learning_rate=0.001):
    """
    Create Encoder-Decoder LSTM for Sequence-to-Sequence Prediction.
    
    This architecture is particularly useful for multi-step ahead forecasting.
    The encoder compresses the input sequence into a context vector, which
    the decoder uses to generate predictions.
    
    Args:
        n_steps (int): Number of time steps in input sequence
        n_features (int): Number of features in each time step
        encoder_units (list): List of units for encoder LSTM layers
        decoder_units (int): Number of units in decoder LSTM
        dense_units (list): List of units for dense layers
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled Encoder-Decoder model
    
    Model Architecture:
        Encoder: Input -> LSTM(128) -> LSTM(64) -> Context Vector
        Decoder: Context -> LSTM(64) -> Dense -> Output(1)
    
    Example:
        >>> model = create_encoder_decoder_lstm()
        >>> model.fit(X_train, y_train)
    """
    
    # Encoder
    encoder_inputs = layers.Input(shape=(n_steps, n_features), name='encoder_input')
    
    # First encoder LSTM
    encoder_lstm1 = layers.LSTM(
        encoder_units[0],
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        name='encoder_lstm_1'
    )
    encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs)
    
    # Second encoder LSTM
    encoder_lstm2 = layers.LSTM(
        encoder_units[1],
        return_state=True,
        dropout=dropout_rate,
        name='encoder_lstm_2'
    )
    _, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)
    
    # Decoder
    decoder_inputs = layers.RepeatVector(1, name='repeat_vector')(state_h2)
    
    decoder_lstm = layers.LSTM(
        decoder_units,
        return_sequences=True,
        dropout=dropout_rate,
        name='decoder_lstm'
    )
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h2, state_c2])
    
    # Dense layers
    x = layers.Flatten(name='flatten')(decoder_outputs)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate * 0.67, name=f'dropout_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(1, name='output')(x)
    
    # Create model
    model = models.Model(encoder_inputs, outputs, name='Encoder_Decoder_LSTM')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_summary(model):
    """
    Get a formatted summary of the model architecture.
    
    Args:
        model (keras.Model): Compiled Keras model
    
    Returns:
        str: Formatted model summary
    """
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    return '\n'.join(summary_list)


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model (keras.Model): Compiled Keras model
    
    Returns:
        dict: Dictionary with parameter counts
    """
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    
    return {
        'total_params': trainable_params + non_trainable_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params
    }


def load_model_with_custom_objects(model_path):
    """
    Load a saved model with custom objects (AttentionLayer).
    
    Args:
        model_path (str): Path to saved model file
    
    Returns:
        keras.Model: Loaded model
    
    Example:
        >>> model = load_model_with_custom_objects('models/bilstm_attention_best.keras')
    """
    custom_objects = {'AttentionLayer': AttentionLayer}
    return keras.models.load_model(model_path, custom_objects=custom_objects)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type='vanilla_lstm', n_steps=12, n_features=6, **kwargs):
    """
    Factory function to create any model by name.
    
    Args:
        model_type (str): Type of model to create
            Options: 'vanilla_lstm', 'bilstm_attention', 'cnn_lstm', 
                    'transformer', 'encoder_decoder'
        n_steps (int): Number of time steps
        n_features (int): Number of features
        **kwargs: Additional arguments for specific model
    
    Returns:
        keras.Model: Compiled model
    
    Example:
        >>> model = create_model('transformer', n_steps=12, n_features=6)
    """
    
    models_dict = {
        'vanilla_lstm': create_vanilla_lstm,
        'bilstm_attention': create_bilstm_attention,
        'cnn_lstm': create_cnn_lstm,
        'transformer': create_transformer_model,
        'encoder_decoder': create_encoder_decoder_lstm
    }
    
    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models_dict.keys())}")
    
    return models_dict[model_type](n_steps=n_steps, n_features=n_features, **kwargs)


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test all model architectures
    """
    print("Testing Model Architectures")
    print("=" * 80)
    
    # Test parameters
    n_steps = 12
    n_features = 6
    batch_size = 32
    
    # Test data
    X_test = np.random.randn(batch_size, n_steps, n_features)
    y_test = np.random.randn(batch_size, 1)
    
    models_to_test = [
        ('Vanilla LSTM', create_vanilla_lstm),
        ('BiLSTM with Attention', create_bilstm_attention),
        ('CNN-LSTM Hybrid', create_cnn_lstm),
        ('Transformer', create_transformer_model),
        ('Encoder-Decoder LSTM', create_encoder_decoder_lstm)
    ]
    
    for name, create_fn in models_to_test:
        print(f"\n{name}")
        print("-" * 80)
        
        # Create model
        model = create_fn(n_steps=n_steps, n_features=n_features)
        
        # Get parameter count
        params = count_parameters(model)
        print(f"Total Parameters: {params['total_params']:,}")
        print(f"Trainable Parameters: {params['trainable_params']:,}")
        
        # Test forward pass
        predictions = model.predict(X_test, verbose=0)
        print(f"Output Shape: {predictions.shape}")
        print(f"Sample Prediction: {predictions[0][0]:.4f}")
        
        # Model summary
        print("\nModel Architecture:")
        model.summary()
    
    print("\n" + "=" * 80)
    print("All models tested successfully!")
