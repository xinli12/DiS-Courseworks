"""Neural network models for MNIST digit addition.

This module contains the model architectures and training utilities for the MNIST
digit addition task. The models are implemented using TensorFlow/Keras with
Optuna for hyperparameter optimization.
"""

import tensorflow as tf
import optuna
from typing import Dict, Any, Tuple, List
import numpy as np
import logging

class DigitAdditionModel:
    """
    A neural network model designed for the adding two MNIST digits task.

    This class implements a fully connected neural network that takes as input 
    a concatenated image of two MNIST digits and predicts their sum. The model 
    leverages a configurable architecture, including hyperparameters for the 
    number of hidden layers, activation functions, dropout, and L2 regularization. 
    The output is a probability distribution over all possible sums (0 through 18).

    Key Features:
    - Fully connected layers for feature extraction and prediction.
    - Configurable hyperparameters to fine-tune model performance.
    - Support for training, evaluation, prediction, and weight management.

    Attributes:
        input_shape (Tuple[int, int]): Shape of the input image, typically 
            (28, 56) for two concatenated 28x28 MNIST digits.
        model (tf.keras.Model): The compiled Keras model, built after 
            calling the `build` method.
        history (tf.keras.callbacks.History): Training history object after 
            model fitting, containing metrics and loss values per epoch.
        hyperparameters (Dict[str, Any]): Dictionary containing model configuration 
            settings, populated during the `build` method.
    """
    
    def __init__(self, input_shape: Tuple[int, int] = (28, 56)):
        """Initialize the model.
        
        Args:
            input_shape: Input image dimensions (height, width)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.hyperparameters = None
        
    def build(self, hyperparameters: Dict[str, Any]) -> None:
        """Build and compile the neural network model.
        
        Args:
            hyperparameters: Dictionary containing:
                - units_per_layer: List of neurons per hidden layer
                - dropout_rate: Dropout probability
                - learning_rate: Learning rate for Adam optimizer
                - l2_reg: L2 regularization factor
                - activation: Activation function name
        """
        self.hyperparameters = hyperparameters
        
        # Input layer
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        
        # Hidden layers
        for units in hyperparameters['units_per_layer']:
            x = tf.keras.layers.Dense(
                units=units,
                activation=hyperparameters['activation'],
                kernel_regularizer=tf.keras.regularizers.l2(hyperparameters['l2_reg'])
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(hyperparameters['dropout_rate'])(x)
            
        # Output layer
        outputs = tf.keras.layers.Dense(
            units=19,  # 19 possible sums (0 to 18)
            activation='softmax'
        )(x)
        
        # Create and compile model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, train_data: tf.data.Dataset, val_data: tf.data.Dataset,
              epochs: int = 50, callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0  # Suppress output for Optuna trials
        )
        return self.history
        
    def evaluate(self, test_data: tf.data.Dataset) -> Tuple[float, float]:
        """Evaluate the model.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Tuple of (test loss, test accuracy)
        """
        return self.model.evaluate(test_data, verbose=0)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            x: Input images
            
        Returns:
            Predicted class probabilities
        """
        return self.model.predict(x, verbose=0)
    
    def save_weights(self, filepath: str) -> None:
        """Save model weights.
        
        Args:
            filepath: Path to save weights file
        """
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath: str) -> None:
        """Load model weights.
        
        Args:
            filepath: Path to weights file
        """
        self.model.load_weights(filepath)


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna.
    
    This class implements hyperparameter optimization for the digit addition model
    using Optuna's efficient search strategies.
    
    Attributes:
        study: Optuna study object
        train_data: Training dataset
        val_data: Validation dataset
        best_trial: Best trial from optimization
        best_model: Best performing model
    """
    
    def __init__(self, train_data: tf.data.Dataset, val_data: tf.data.Dataset, 
                 param_ranges: Dict[str, Any] = None):
        """Initialize the optimizer.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            param_ranges: Dictionary defining hyperparameter search ranges with keys:
                - n_layers: Tuple[int, int] for (min_layers, max_layers)
                - units_per_layer: Tuple[int, int] for (min_units, max_units)
                - dropout_rate: Tuple[float, float] for (min_rate, max_rate)
                - learning_rate: Tuple[float, float] for (min_lr, max_lr)
                - l2_reg: Tuple[float, float] for (min_l2, max_l2)
                - activation: List[str] of activation function names
                If None, uses default ranges.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.study = None
        self.best_trial = None
        self.best_model = None
        
        # Default hyperparameter ranges
        self.param_ranges = {
            'n_layers': (1, 3),
            'units_per_layer': (32, 512),
            'dropout_rate': (0.1, 0.5),
            'learning_rate': (1e-5, 1e-2),
            'l2_reg': (1e-6, 1e-3),
            'activation': ['relu', 'elu']
        }
        
        # Update with user-provided ranges
        if param_ranges is not None:
            self.param_ranges.update(param_ranges)
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy
        """
        # Define hyperparameter search space using ranges
        hyperparameters = {
            'n_layers': trial.suggest_int('n_layers', 
                                        self.param_ranges['n_layers'][0],
                                        self.param_ranges['n_layers'][1]),
            'units_per_layer': [],
            'dropout_rate': trial.suggest_float('dropout_rate',
                                              self.param_ranges['dropout_rate'][0],
                                              self.param_ranges['dropout_rate'][1]),
            'learning_rate': trial.suggest_float('learning_rate',
                                               self.param_ranges['learning_rate'][0],
                                               self.param_ranges['learning_rate'][1],
                                               log=True),
            'l2_reg': trial.suggest_float('l2_reg',
                                        self.param_ranges['l2_reg'][0],
                                        self.param_ranges['l2_reg'][1],
                                        log=True),
            'activation': trial.suggest_categorical('activation',
                                                 self.param_ranges['activation'])
        }
        
        # Define units for each layer
        for i in range(hyperparameters['n_layers']):
            units = trial.suggest_int(f'units_l{i}',
                                    self.param_ranges['units_per_layer'][0],
                                    self.param_ranges['units_per_layer'][1],
                                    log=True)
            hyperparameters['units_per_layer'].append(units)
            
        # Create and train model
        model = DigitAdditionModel()
        model.build(hyperparameters)
        
        # Use early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.train(
            self.train_data,
            self.val_data,
            epochs=50,
            callbacks=[early_stopping]
        )
        
        # Get best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])
        
        # Store best model if this is the best trial so far
        if self.best_trial is None or val_accuracy > self.study.trials[self.best_trial.number].value:
            self.best_trial = trial
            self.best_model = model
            
        return val_accuracy
    
    def optimize(self, n_trials: int = 100, study_name: str = "mnist_addition", 
                log_file: str = None) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            log_file: Path to save optimization log. If None, prints to console.
            
        Returns:
            Dictionary of best hyperparameters
        """
        # Set up logging
        if log_file:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s'
            )
            
        # Log hyperparameter ranges
        logging.info("Starting hyperparameter optimization")
        logging.info("Parameter ranges:")
        for param, range_val in self.param_ranges.items():
            logging.info(f"  {param}: {range_val}")
            
        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        def log_trial(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            """Callback to log trial results."""
            if trial.state == optuna.trial.TrialState.COMPLETE:
                logging.info(
                    f"\nTrial {trial.number} finished with value: {trial.value:.4f}"
                )
                logging.info("Parameters:")
                for param, value in trial.params.items():
                    logging.info(f"  {param}: {value}")
                    
                # Log if this is the best trial so far
                if study.best_trial == trial:
                    logging.info("  This is the best trial so far!")
                    
        # Run optimization
        logging.info(f"\nStarting {n_trials} trials...")
        self.study.optimize(self.objective, n_trials=n_trials, callbacks=[log_trial])
        
        # Log final results
        logging.info("\nOptimization finished!")
        logging.info(f"Best trial value (validation accuracy): {self.study.best_value:.4f}")
        logging.info("Best parameters:")
        for param, value in self.study.best_params.items():
            logging.info(f"  {param}: {value}")
            
        return self.study.best_params
    
    def get_results_summary(self) -> str:
        """Get a formatted summary of optimization results.
        
        Returns:
            String containing optimization results summary
        """
        if self.study is None:
            return "No optimization results available. Run optimize() first."
            
        summary = "Hyperparameter Optimization Results:\n"
        summary += "-" * 50 + "\n\n"
        
        summary += "Best Trial:\n"
        summary += f"  Value (Validation Accuracy): {self.study.best_value:.4f}\n"
        summary += "  Hyperparameters:\n"
        for param, value in self.study.best_params.items():
            summary += f"    {param}: {value}\n"
            
        summary += "\nParameter Importance:\n"
        importances = optuna.importance.get_param_importances(self.study)
        for param, importance in importances.items():
            summary += f"  {param}: {importance:.4f}\n"
            
        return summary