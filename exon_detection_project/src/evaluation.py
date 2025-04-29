# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, classification_report
import os

class ModelEvaluator:
    def __init__(self, results_dir='results'):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        results_dir : str
            Directory to save evaluation results
        """
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        self.metrics_dir = os.path.join(results_dir, 'metrics')
        
        # Create directories if they don't exist
        for directory in [self.figures_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def evaluate_model(self, model, X_test, y_test, threshold=0.3):
        y_pred_prob = model.predict(X_test)
        y_test_flat = y_test.flatten()
        y_pred_prob_flat = y_pred_prob.flatten()
        y_pred_flat = (y_pred_prob_flat >= threshold).astype(int)
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, classification_report
        fpr, tpr, _ = roc_curve(y_test_flat, y_pred_prob_flat)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test_flat, y_pred_prob_flat)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(y_test_flat, y_pred_flat)
        report = classification_report(y_test_flat, y_pred_flat, output_dict=True)
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        return metrics
    
    def plot_roc_curve(self, metrics, save=True):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_pr_curve(self, metrics, save=True):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(metrics['recall'], metrics['precision'], color='blue', lw=2, 
                 label=f'PR curve (AUC = {metrics["pr_auc"]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, metrics, save=True):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Exon', 'Exon'],
                    yticklabels=['Non-Exon', 'Exon'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_training_history(self, history, save=True):
        """
        Plot training history.
        
        Parameters:
        -----------
        history : tf.keras.callbacks.History
            Training history
        save : bool
            Whether to save the plot
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_metrics(self, metrics):
        """Save metrics to file"""
        # Save classification report
        with open(os.path.join(self.metrics_dir, 'classification_report.txt'), 'w') as f:
            for class_name, values in metrics['classification_report'].items():
                if not isinstance(values, dict):
                    continue
                f.write(f"{class_name}:\n")
                for metric, value in values.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        # Save AUC values
        with open(os.path.join(self.metrics_dir, 'auc_values.txt'), 'w') as f:
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
    
    def visualize_predictions(self, X_test, y_test, y_pred, indices=None, save=True):
        """
        Visualize model predictions.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test input data
        y_test : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        indices : list, optional
            Indices of samples to visualize
        save : bool
            Whether to save the plot
        """
        if indices is None:
            # Randomly select 5 samples
            indices = np.random.choice(len(y_test), 5, replace=False)
        
        fig, axes = plt.subplots(len(indices), 1, figsize=(15, 3*len(indices)))
        
        for i, idx in enumerate(indices):
            ax = axes[i] if len(indices) > 1 else axes
            
            # Get the sample
            true_labels = y_test[idx]
            pred_probs = y_pred[idx]
            
            # Plot
            ax.plot(true_labels, 'g-', linewidth=2, label='True')
            ax.plot(pred_probs, 'b-', linewidth=2, alpha=0.7, label='Predicted')
            ax.set_title(f'Sample {idx}')
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel('Position')
            ax.set_ylabel('Exon Probability')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'prediction_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_sample_sequence(self, X_test, y_test, y_pred, sample_idx=0, save=True):
        """
        Visualize a single sample sequence with its prediction.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test input data
        y_test : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted probabilities
        sample_idx : int
            Index of the sample to visualize
        save : bool
            Whether to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Get the sample
        sample = X_test[sample_idx]
        true_labels = y_test[sample_idx]
        pred_probs = y_pred[sample_idx]
        
        # Plot DNA sequence (convert one-hot back to nucleotides)
        nucleotides = np.argmax(sample, axis=1)
        nuc_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        seq = ''.join([nuc_map[n] for n in nucleotides])
        
        # Plot exon predictions
        axes[0].plot(true_labels, 'g-', linewidth=2, label='True Exons')
        axes[0].set_title(f'Exon Positions for Sample {sample_idx}')
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].set_ylabel('Is Exon')
        axes[0].legend()
        
        # Plot predicted probabilities
        axes[1].plot(pred_probs, 'b-', linewidth=2, label='Predicted Probabilities')
        axes[1].set_title(f'Predicted Exon Probabilities')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        
        # Display sequence visualization
        axes[2].imshow(sample.T, aspect='auto', cmap='viridis')
        axes[2].set_title('Sequence One-Hot Encoding')
        axes[2].set_xlabel('Position')
        axes[2].set_ylabel('Nucleotide (A,C,G,T)')
        axes[2].set_yticks([0, 1, 2, 3])
        axes[2].set_yticklabels(['A', 'C', 'G', 'T'])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, f'sample_{sample_idx}_visualization.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()