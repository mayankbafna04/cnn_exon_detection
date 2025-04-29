# src/cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class CNNModel:
    def __init__(self, input_shape, hyperparameters=None):
        self.input_shape = input_shape
        self.hyperparameters = {
            'filters1': 64,
            'filters2': 128,
            'filters3': 256,
            'kernel_size1': 7,
            'kernel_size2': 5,
            'kernel_size3': 3,
            'pool_size': 2,
            'dropout_rate': 0.25,
            'dense_units': 128,
            'learning_rate': 0.001,
            'l2_reg': 1e-6
        }
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.model = self._build_model()

    def _build_model(self):
        hp = self.hyperparameters
        l2 = regularizers.l2(hp['l2_reg'])
        model = models.Sequential([
        layers.Input(shape=self.input_shape),
        # First conv block
        layers.Conv1D(hp['filters1'], hp['kernel_size1'], activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=hp['pool_size']),
        layers.Dropout(hp['dropout_rate']),
        
        # Second conv block
        layers.Conv1D(hp['filters2'], hp['kernel_size2'], activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=hp['pool_size']),
        layers.Dropout(hp['dropout_rate']),
        
        # Third conv block (added)
        layers.Conv1D(hp['filters3'], hp['kernel_size3'], activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=hp['pool_size']),
        layers.Dropout(hp['dropout_rate']),
        
            # Dense
            layers.Flatten(),
            layers.Dense(hp['dense_units'], activation='relu', kernel_regularizer=l2),
            layers.BatchNormalization(),
            layers.Dropout(hp['dropout_rate']),
            layers.Dense(self.input_shape[0], activation='sigmoid')
        ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp['learning_rate'],
            decay_steps=10000, decay_rate=0.96)
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, model_path=None, class_weight=None):
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        if model_path:
            callbacks.append(ModelCheckpoint(filepath=model_path, monitor='val_loss' if X_val is not None else 'loss', save_best_only=True))
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight
        )
        return history
    
    def predict_with_threshold(self, X, threshold=0.7, early_stop=True):
        """
        Predict with early stopping when high confidence exons are found.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        threshold : float
            Confidence threshold
        early_stop : bool
            Whether to stop processing once a high confidence exon is found
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        predictions = self.model.predict(X)
        
        if early_stop:
            # For each sequence, check if there's a high confidence exon
            high_conf_mask = np.max(predictions, axis=1) > threshold
            
            # For sequences with high confidence predictions, we can 
            # avoid further processing in downstream tasks
            return predictions, high_conf_mask
        return predictions

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = models.load_model(path)
