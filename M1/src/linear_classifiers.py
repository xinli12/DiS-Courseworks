import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .data_handler import MNISTAdditionDataset

class LinearClassifier:
    """A class implementing linear classifiers for MNIST addition task.
    
    This class provides two approaches to classify pairs of MNIST digits and predict their sum:
    1. Joint classification: treats concatenated digit images as a single input
    2. Sequential classification: classifies each digit separately and sums the results
    
    The class uses logistic regression as the base classifier.
    """
    
    def __init__(self, seed: int = 42):
        """
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.dataset = MNISTAdditionDataset(seed=seed)
        
    def train_joint_classifier(self, train_size: int) -> Tuple[LogisticRegression, float]:
        """Train a single classifier on concatenated images.
        
        Args:
            train_size (int): Number of training samples to use
            
        Returns:
            Tuple containing:
            - Trained classifier
            - Test accuracy
        """
        # Get data
        data = self.dataset.create_datasets(train_size=train_size)
        x_train, y_train = data['train'][0], data['train'][1]
        x_test, y_test = data['test'][0], data['test'][1]
        
        # Reshape data
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        
        # Train classifier
        classifier = LogisticRegression(multi_class='multinomial', 
                                      penalty='l2', 
                                      solver='sag',
                                      tol=0.01,
                                      random_state=self.seed)
        classifier.fit(x_train_flat, y_train)
        
        # Evaluate
        y_pred = classifier.predict(x_test_flat)
        y_pred_proba = classifier.predict_proba(x_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        
        return classifier, accuracy, y_pred_proba
    
    def train_sequential_classifier(self, train_size: int) -> Tuple[LogisticRegression, float]:
        """Train a single classifier that will be applied sequentially on each digit.
        
        Args:
            train_size (int): Number of training samples to use
            
        Returns:
            Tuple containing:
            - Trained classifier
            - Test accuracy
        """
        # Get data
        data = self.dataset.create_datasets(train_size=train_size)
        x_train, y_train = data['train'][0], data['train'][1]
        x_test, y_test = data['test'][0], data['test'][1]
        
        # Split images into left and right digits
        x_train_left = x_train[:, :, :28].reshape(x_train.shape[0], -1)
        x_train_right = x_train[:, :, 28:].reshape(x_train.shape[0], -1)
        x_test_left = x_test[:, :, :28].reshape(x_test.shape[0], -1)
        x_test_right = x_test[:, :, 28:].reshape(x_test.shape[0], -1)
        
        # Train classifier on individual digits
        classifier = LogisticRegression(multi_class='multinomial',
                                      solver='sag',
                                      tol=0.01,
                                      random_state=self.seed)
        
        # Get individual digit labels
        _, _, digit_labels_train = data['train']
        _, _, digit_labels_test = data['test']
        
        # Combine left and right digits for training
        x_train_combined = np.vstack([x_train_left, x_train_right])
        y_train_combined = np.concatenate([digit_labels_train[:, 0], digit_labels_train[:, 1]])
        
        # Train on individual digits
        classifier.fit(x_train_combined, y_train_combined)
        
        # Predict individual digits and sum them
        left_pred = classifier.predict(x_test_left)
        left_pred_proba = classifier.predict_proba(x_test_left)
        right_pred = classifier.predict(x_test_right)
        right_pred_proba = classifier.predict_proba(x_test_right)
        y_pred = left_pred + right_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        return classifier, accuracy, (left_pred_proba, right_pred_proba)
    
    def train_sequential_probabilistic(self, train_size: int) -> Tuple[LogisticRegression, float]:
        """Train a classifier that uses probability distributions over individual digits to predict sums.
        
        This method trains a single digit classifier and uses probability combinations to predict sums.
        For each possible sum (0-18), it calculates the total probability by considering all possible
        digit combinations that could result in that sum.
        
        Args:
            train_size (int): Number of training samples to use
            
        Returns:
            Tuple containing:
            - Trained classifier
            - Test accuracy
        """
        # Get data
        data = self.dataset.create_datasets(train_size=train_size)
        x_train, y_train = data['train'][0], data['train'][1]
        x_test, y_test = data['test'][0], data['test'][1]
        
        # Split images and prepare training data
        x_train_left = x_train[:, :, :28].reshape(x_train.shape[0], -1)
        x_train_right = x_train[:, :, 28:].reshape(x_train.shape[0], -1)
        x_test_left = x_test[:, :, :28].reshape(x_test.shape[0], -1)
        x_test_right = x_test[:, :, 28:].reshape(x_test.shape[0], -1)
        
        # Train classifier on individual digits
        classifier = LogisticRegression(multi_class='multinomial',
                                    solver='sag',
                                    tol=0.01,
                                    random_state=self.seed)
        
        # Get individual digit labels and combine data for training
        _, _, digit_labels_train = data['train']
        x_train_combined = np.vstack([x_train_left, x_train_right])
        y_train_combined = np.concatenate([digit_labels_train[:, 0], digit_labels_train[:, 1]])
        
        # Train on individual digits
        classifier.fit(x_train_combined, y_train_combined)
        
        # Get probabilities for left and right digits
        left_probs = classifier.predict_proba(x_test_left)
        right_probs = classifier.predict_proba(x_test_right)
        
        # Calculate sum probabilities
        n_samples = x_test.shape[0]
        sum_probs = np.zeros((n_samples, 19))  # 19 possible sums (0-18)
        
        for i in range(n_samples):
            for sum_val in range(19):
                prob = 0
                for d1 in range(10):
                    for d2 in range(10):
                        if d1 + d2 == sum_val:
                            prob += left_probs[i][d1] * right_probs[i][d2]
                sum_probs[i, sum_val] = prob
        
        # Get predictions and calculate accuracy
        y_pred = np.argmax(sum_probs, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        return classifier, accuracy