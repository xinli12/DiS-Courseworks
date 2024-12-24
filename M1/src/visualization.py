import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def plot_sample_additions(x: np.ndarray, y: np.ndarray, digit_labels: np.ndarray, num_samples: int = 5):
    """Plot sample digit addition examples with labels and equations.
    
    Args:
        x (np.ndarray): Combined images of shape (N, 28, 56)
        y (np.ndarray): Sum labels of shape (N,)
        digit_labels (np.ndarray): Individual digit labels of shape (N, 2)
        num_samples (int, optional): Number of examples to plot. Defaults to 5.
        
    Returns:
        matplotlib.figure.Figure: Figure containing the plotted examples
        
    Note:
        - Each row shows one example
        - Individual digit labels are shown above each digit
        - Complete equation is shown on the right
        - Images are displayed in grayscale
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(x), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Plot the combined image
        axes[i].imshow(x[idx], cmap='gray', interpolation='nearest')
        
        # Add individual digit labels above the numbers
        axes[i].text(14, -2, str(digit_labels[idx][0]), 
                    horizontalalignment='center', fontsize=11,
                    fontfamily='serif', usetex=True)
        axes[i].text(42, -2, str(digit_labels[idx][1]), 
                    horizontalalignment='center', fontsize=11,
                    fontfamily='serif', usetex=True)
        
        # Add equation on the right
        equation = f"${digit_labels[idx][0]} + {digit_labels[idx][1]} = {y[idx]}$"
        axes[i].text(70, 14, equation, fontsize=11,
                    fontfamily='serif', usetex=True)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_history(history):
    """Plot training history showing loss and MAE curves.
    
    Args:
        history: Keras history object containing training metrics
        
    Returns:
        matplotlib.figure.Figure: Figure containing two subplots:
            - Left: Training and validation loss
            - Right: Training and validation MAE
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_test_classes, y_pred_classes, verbose: bool = False):
    """Plot and display confusion matrix.
    
    Args:
        y_test_classes: True class labels
        y_pred_classes: Predicted class labels
        verbose: Whether to print the classification report


    """
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(False)
    
    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes))

    plt.tight_layout()
    return fig
    

def visualize_incorrect_predictions(x_test, y_pred_classes, y_test_classes, digit_labels_test, num_samples: int = 3):
    """Visualize examples of incorrect predictions.
    
    Args:
        x_test: Test input data
        y_pred_classes: Predicted class labels
        y_test_classes: True class labels
        digit_labels_test: Original digit labels for visualization
        num_samples: Number of incorrect predictions to visualize


    """
    incorrect_mask = y_pred_classes != y_test_classes
    incorrect_predictions = x_test[incorrect_mask]
    incorrect_true = y_test_classes[incorrect_mask]
    incorrect_pred = y_pred_classes[incorrect_mask]
    
    if len(incorrect_predictions) > 0:
        num_samples = min(num_samples, len(incorrect_predictions))
        fig = plot_sample_additions(
            incorrect_predictions, 
            incorrect_pred,
            digit_labels_test[incorrect_mask],
            num_samples=num_samples
        )
    else:
        print("No incorrect predictions found!")

    return fig
