# src/pso_optimizer.py
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .cnn_model import CNNModel

class Particle:
    def __init__(self, dimensions, bounds):
        """
        Initialize a particle for PSO.
        
        Parameters:
        -----------
        dimensions : int
            Number of dimensions (hyperparameters)
        bounds : list of tuples
            Bounds for each dimension (min, max)
        """
        # Initialize position and velocity
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dimensions)])
        
        # Initialize best position and score
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')  # For minimization problem
        self.current_score = float('inf')

class PSOOptimizer:
    def __init__(self, n_particles=10, dimensions=9, bounds=None, w=0.7, c1=1.5, c2=1.5, max_iter=15):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds if bounds else self._set_default_bounds()
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.particles = [Particle(dimensions, self.bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.param_names = [
            'filters1', 'filters2', 'filters3',
            'kernel_size1', 'kernel_size2', 'kernel_size3',
            'pool_size', 'dropout_rate', 'dense_units'
        ]

    def _set_default_bounds(self):
        return [
            (32, 128),   # filters1
            (64, 256),   # filters2
            (128, 384),  # filters3
            (3, 11),     # kernel_size1
            (3, 9),      # kernel_size2
            (3, 7),      # kernel_size3
            (2, 4),      # pool_size
            (0.1, 0.5),  # dropout_rate
            (64, 256)    # dense_units
        ]
    
    def _position_to_hyperparameters(self, position):
        """Convert particle position to CNN hyperparameters"""
        # Convert to appropriate types (int for most, float for dropout)
        hyperparameters = {}
        
        for i, name in enumerate(self.param_names):
            # Integer hyperparameters
            if name != 'dropout_rate':
                hyperparameters[name] = int(position[i])
            # Float hyperparameters
            else:
                hyperparameters[name] = float(position[i])
        
        return hyperparameters
    
    # In src/pso_optimizer.py
    def _evaluate_particle(self, particle, X_train, y_train, X_val, y_val, input_shape):
        """
        Evaluate a particle by training CNN with its hyperparameters.
        
        Parameters:
        -----------
        particle : Particle
            Particle to evaluate
        X_train : np.ndarray
            Training input data
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation input data
        y_val : np.ndarray
            Validation labels
        input_shape : tuple
            Shape of input data
            
        Returns:
        --------
        float
            Validation loss (score)
        """
        # Convert particle position to hyperparameters
        hyperparameters = self._position_to_hyperparameters(particle.position)
        
        # Create and train a CNN model with these hyperparameters
        model = CNNModel(input_shape, hyperparameters)
        
        # Train the model (removed the callbacks parameter)
        history = model.train(
            X_train, y_train, 
            X_val=X_val, 
            y_val=y_val, 
            batch_size=32, 
            epochs=15  # Fewer epochs for faster optimization
        )
        
        # Get the best validation loss
        val_loss = min(history.history['val_loss'])
        
        return val_loss

    
    def optimize(self, X_train, y_train, input_shape):
        """
        Optimize CNN hyperparameters using PSO.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input data
        y_train : np.ndarray
            Training labels
        input_shape : tuple
            Shape of input data
            
        Returns:
        --------
        dict
            Optimized hyperparameters
        float
            Best score achieved
        """
        # Split training data to create a validation set
        X_train_pso, X_val_pso, y_train_pso, y_val_pso = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print("Starting PSO optimization...")
        
        # Main PSO loop
        for iteration in tqdm(range(self.max_iter)):
            # Evaluate each particle
            for particle in self.particles:
                # Skip evaluation if out of bounds
                if any(particle.position[i] < self.bounds[i][0] or 
                       particle.position[i] > self.bounds[i][1] 
                       for i in range(self.dimensions)):
                    continue
                
                # Evaluate the particle
                score = self._evaluate_particle(
                    particle, X_train_pso, y_train_pso, X_val_pso, y_val_pso, input_shape
                )
                
                particle.current_score = score
                
                # Update particle best if current position is better
                if score < particle.best_score:
                    particle.best_position = np.copy(particle.position)
                    particle.best_score = score
                    
                    # Update global best if this particle is the best so far
                    if score < self.global_best_score:
                        self.global_best_position = np.copy(particle.position)
                        self.global_best_score = score
                        print(f"New best score: {score:.4f}")
                        print(f"Parameters: {self._position_to_hyperparameters(particle.position)}")
            
            # Update particle velocities and positions
            for particle in self.particles:
                # Cognitive component (personal best)
                cognitive = self.c1 * random.random() * (particle.best_position - particle.position)
                
                # Social component (global best)
                social = self.c2 * random.random() * (self.global_best_position - particle.position)
                
                # Update velocity
                particle.velocity = self.w * particle.velocity + cognitive + social
                
                # Update position
                particle.position = particle.position + particle.velocity
                
                # Ensure position stays within bounds
                for i in range(self.dimensions):
                    if particle.position[i] < self.bounds[i][0]:
                        particle.position[i] = self.bounds[i][0]
                    elif particle.position[i] > self.bounds[i][1]:
                        particle.position[i] = self.bounds[i][1]
        
        # Convert best position to hyperparameters
        best_hyperparameters = self._position_to_hyperparameters(self.global_best_position)
        
        print(f"PSO optimization completed!")
        print(f"Best score: {self.global_best_score:.4f}")
        print(f"Best hyperparameters: {best_hyperparameters}")
        
        return best_hyperparameters, self.global_best_score
