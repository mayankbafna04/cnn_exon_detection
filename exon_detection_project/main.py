import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.data_processing import DNADataProcessor
from src.cnn_model import CNNModel
from src.pso_optimizer import PSOOptimizer
from src.evaluation import ModelEvaluator
from sklearn.utils.class_weight import compute_class_weight
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Exon Detection using CNN and PSO')
    parser.add_argument('--email', type=str, default="your.email@example.com",
                        help="Email for Entrez services")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from latest checkpoint")
    parser.add_argument('--skip-pso', action='store_true',
                        help="Skip PSO optimization and use default hyperparameters")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Training batch size")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Threshold for exon detection")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # List of accession numbers
    accession_numbers = [
        # ... (your accession numbers here)
    ]

    # Step 1: Download and preprocess data
    data_processor = DNADataProcessor(accession_numbers=accession_numbers, email=args.email)
    print("\nStep 1: Downloading sequences from NCBI...")
    data_processor.download_sequences()

    print("\nStep 2: Processing DNA sequences...")
    (X_train, y_train), (X_test, y_test) = data_processor.split_train_test()
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")

    # Step 3: Optimize hyperparameters using PSO
    if not args.skip_pso:
        print("\nStep 3: Optimizing CNN hyperparameters using PSO...")
        pso_optimizer = PSOOptimizer(n_particles=10, dimensions=9, max_iter=15)
        best_hyperparameters, best_score = pso_optimizer.optimize(X_train, y_train, input_shape)
    else:
        print("\nSkipping PSO, using default hyperparameters.")
        best_hyperparameters = {
            'num_filters': 64,
            'kernel_size': 3,
            'pool_size': 2,
            'dense_units': 128,
            'dropout_rate': 0.5,
            'optimizer': 'adam',
            'activation': 'relu',
            'batch_norm': True,
            'num_conv_layers': 2
        }
        best_score = 0.0

    # Step 4: Training with checkpointing and early stopping
    print("\nStep 4: Training the CNN model with optimized hyperparameters...")

    cnn_model = CNNModel(input_shape, best_hyperparameters)

    model_path = "models/saved_models/best_model.h5"
    checkpoint_dir = "models/checkpoints"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    initial_epoch = 0
    if args.resume:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"\nResuming training from checkpoint: {latest_checkpoint}")
            cnn_model.model.load_weights(latest_checkpoint)
            try:
                initial_epoch = int(os.path.basename(latest_checkpoint).split('-')[1])
            except:
                print("Could not determine epoch from checkpoint filename. Starting from 0.")
        else:
            print("No checkpoint found. Starting from scratch.")
    else:
        print("\nStarting training from scratch...")

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model-{epoch:04d}-{val_loss:.4f}"),
        save_weights_only=True,
        save_best_only=False,
        verbose=1,
        save_freq='epoch'
    )

    best_model_callback = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Compute class weights
    y_train_flat = y_train.flatten()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flat), y=y_train_flat)
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\nTraining from epoch {initial_epoch}...")
    history = cnn_model.model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_callback, best_model_callback, early_stopping]
    )

    # Step 5: Evaluation
    print("\nStep 5: Evaluating the model...")
    evaluator = ModelEvaluator()
    y_pred = cnn_model.predict(X_test)

    metrics = evaluator.evaluate_model(cnn_model, X_test, y_test, threshold=args.threshold)
    evaluator.plot_roc_curve(metrics)
    evaluator.plot_pr_curve(metrics)
    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_training_history(history)
    evaluator.visualize_predictions(X_test, y_test, y_pred)
    evaluator.save_metrics(metrics)

    # Final results
    print("\nExon detection project completed successfully!")
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best validation loss: {best_score:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR AUC: {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
