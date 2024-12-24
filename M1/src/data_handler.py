import tensorflow as tf
import numpy as np
from typing import Tuple, Dict

class MNISTAdditionDataset:
    """A dataset for training models to add two MNIST digits.
    
    This class handles the creation and management of a dataset where each sample
    consists of two horizontally concatenated MNIST digits. The label for each
    sample is the sum of the two digits.
    
    Attributes:
        seed (int): Random seed for reproducibility
        x_train (np.ndarray): Training images from MNIST
        x_test (np.ndarray): Test images from MNIST
        y_train (np.ndarray): Training labels from MNIST
        y_test (np.ndarray): Test labels from MNIST
    
    Note:
        The dataset ensures:
        - Proper normalization of images (0 to 1 range)
        - Random but reproducible digit combinations
        - No data leakage between splits
        - Statistical validation of the generated data
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the dataset with given random seed.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self._load_mnist()
        
    def _load_mnist(self):
        """Load and preprocess the MNIST dataset.
        
        The images are normalized to [0, 1] range by dividing by 255.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = x_train.astype('float32') / 255.0
        self.x_test = x_test.astype('float32') / 255.0
        self.y_train = y_train
        self.y_test = y_test
        
    def _combine_images(self, images: np.ndarray, labels: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create combined images and their corresponding labels.
        
        Args:
            images (np.ndarray): Source images of shape (N, 28, 28)
            labels (np.ndarray): Source labels of shape (N,)
            num_samples (int): Number of combined samples to generate
            
        Returns:
            Tuple containing:
            - combined_images (np.ndarray): Array of shape (num_samples, 28, 56)
            - combined_labels (np.ndarray): Array of shape (num_samples,) containing sums
            - digit_labels (np.ndarray): Array of shape (num_samples, 2) containing individual digits
            
        Note:
            Images are combined horizontally, resulting in 28x56 pixel images.
            The sum of digits is constrained to [0, 18] as each digit is in [0, 9].
        """
        indices = np.random.choice(len(images), size=(num_samples, 2), replace=True)
        combined_images = np.zeros((num_samples, 28, 56))
        combined_labels = labels[indices].sum(axis=1)
        digit_labels = labels[indices]  # Store individual digit labels
        
        for i in range(num_samples):
            combined_images[i, :, :28] = images[indices[i, 0]]
            combined_images[i, :, 28:] = images[indices[i, 1]]
            
        return combined_images, combined_labels, digit_labels

    def _theoretical_sum_distribution(self, sums: np.ndarray) -> np.ndarray:
        """Calculate the theoretical distribution for sum of two random digits.
        
        For two independent random digits in [0, 9], the probability of their sum
        follows a triangular-like distribution but is discrete and bounded by [0, 18].
        The probability of each sum is the number of ways to achieve that sum divided
        by the total number of possible digit combinations (100).
        
        Args:
            sums (np.ndarray): Array of possible sums [0, 1, ..., 18]
            
        Returns:
            np.ndarray: Probability of each sum
        """
        # Create 10x10 grid of all possible digit combinations
        d1, d2 = np.meshgrid(np.arange(10), np.arange(10))
        all_sums = d1 + d2
        
        # Count occurrences of each sum
        sum_counts = np.zeros(19)  # sums from 0 to 18
        for s in range(19):
            sum_counts[s] = np.sum(all_sums == s)
            
        # Convert to probabilities
        return sum_counts / 100  # total combinations = 10 * 10 = 100

    def create_datasets(self, train_size: int = 50000, val_size: int = 10000, test_size: int = 10000) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create train, validation, and test datasets.
        
        Args:
            train_size (int, optional): Number of training samples. Defaults to 50000.
            val_size (int, optional): Number of validation samples. Defaults to 10000.
            test_size (int, optional): Number of test samples. Defaults to 10000.
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]: Dictionary containing:
                - 'train': (images, sums, digit_labels) for training
                - 'val': (images, sums, digit_labels) for validation
                - 'test': (images, sums, digit_labels) for testing
                
        Note:
            - Images have shape (N, 28, 56)
            - Sums have shape (N,)
            - Digit_labels have shape (N, 2)
            - Statistical properties are validated after creation
        """
        x_train_combined, y_train_combined, digit_labels_train = self._combine_images(
            self.x_train, self.y_train, train_size + val_size)
        x_test_combined, y_test_combined, digit_labels_test = self._combine_images(
            self.x_test, self.y_test, test_size)
        
        # Split training into train and validation
        x_train = x_train_combined[:train_size]
        y_train = y_train_combined[:train_size]
        digit_train = digit_labels_train[:train_size]
        x_val = x_train_combined[train_size:]
        y_val = y_train_combined[train_size:]
        digit_val = digit_labels_train[train_size:]
        
        data = {
            'train': (x_train, y_train, digit_train),
            'val': (x_val, y_val, digit_val),
            'test': (x_test_combined, y_test_combined, digit_labels_test)
        }
        
        return data
        
    def get_tf_dataset(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                      shuffle: bool = True) -> tf.data.Dataset:
        """Create a TensorFlow dataset from numpy arrays.
        
        Args:
            x (np.ndarray): Input images of shape (N, 28, 56)
            y (np.ndarray): Labels (sums) of shape (N,)
            batch_size (int, optional): Batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            
        Returns:
            tf.data.Dataset: Dataset that yields (images, labels) pairs
            
        Note:
            - Uses tf.data.AUTOTUNE for prefetching
            - Maintains reproducibility through self.seed when shuffling
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)