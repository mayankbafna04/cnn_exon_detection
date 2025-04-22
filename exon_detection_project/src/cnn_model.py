# src/cnn_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class CNNModel:
    def __init__(self, input_shape, hyperparameters=None):
        """
        Initialize the CNN model for exon detection.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (window_size, 4)
        hyperparameters : dict, optional
            Dictionary containing hyperparameters
        """
        self.input_shape = input_shape
        
        # Default hyperparameters
        self.hyperparameters = {
            'filters1': 64,
            'filters2': 128,
            'kernel_size1': 7,
            'kernel_size2': 5,
            'pool_size': 2,
            'dropout_rate': 0.25,
            'dense_units': 128,
            'learning_rate': 0.001
        }
        
        # Update with custom hyperparameters if provided
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the CNN model"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv1D(
                filters=self.hyperparameters['filters1'],
                kernel_size=self.hyperparameters['kernel_size1'],
                activation='relu',
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=self.hyperparameters['pool_size']),
            layers.Dropout(self.hyperparameters['dropout_rate']),
            
            # Second convolutional block
            layers.Conv1D(
                filters=self.hyperparameters['filters2'],
                kernel_size=self.hyperparameters['kernel_size2'],
                activation='relu',
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=self.hyperparameters['pool_size']),
            layers.Dropout(self.hyperparameters['dropout_rate']),
            
            # Dense layer
            layers.Flatten(),
            layers.Dense(self.hyperparameters['dense_units'], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.hyperparameters['dropout_rate']),
            
            # Output layer - reshape to match the window size
            layers.Dense(self.input_shape[0], activation='sigmoid')
        ])
        
        # Compile the model
        optimizer = optimizers.Adam(learning_rate=self.hyperparameters['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, model_path=None):
        """
        Train the CNN model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation input data
        y_val : np.ndarray, optional
            Validation labels
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        model_path : str, optional
            Path to save the best model
            
        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        callbacks = []
        
        # Add early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True
        ))
        
        # Add model checkpoint if path is provided
        if model_path:
            callbacks.append(ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ))
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with the model"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        return self.model.evaluate(X_test, y_test)
    
    def save(self, path):
        """Save the model to disk"""
        self.model.save(path)
    
    def load(self, path):
        """Load the model from disk"""
        self.model = models.load_model(path)
