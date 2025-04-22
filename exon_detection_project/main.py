# main.py
import os
import numpy as np
import tensorflow as tf
from src.data_processing import DNADataProcessor
from src.cnn_model import CNNModel
from src.pso_optimizer import PSOOptimizer
from src.evaluation import ModelEvaluator
import matplotlib.pyplot as plt

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # List of accession numbers
    accession_numbers = [
        "Z49258", "X73428", "X66114", "X14487", "X64467", "X01392", "X04898", 
        "X01038", "X05151", "X52150", "X63600", "X68793", "X69907", "X72861", 
        "X12706", "X04143", "Y00081", "X54486", "X52889", "X06882", "X14974", 
        "X06180", "X15334", "X57152", "Z26491", "X62891", "X52851", "X14720", 
        "Z18859", "Z46254", "X68303", "X02612", "X78212", "X84707", "X15215", 
        "X79198", "Z48950", "X76776", "X02882", "X00492", "X61755", "Y00371", 
        "X60459", "V00536", "Z00010", "X00695", "X03833", "X04500", "X64532", 
        "V00565", "X03072", "X14445", "X52138", "X04981", "X62654", "Y00477", 
        "X54489", "Z14977", "Z48051", "Z33457", "Z29373", "Y00067", "X16277", 
        "X74614", "X54156"
    ]
    
    # Initialize data processor
    data_processor = DNADataProcessor(
        accession_numbers=accession_numbers,
        email="your.email@example.com"  # Replace with your email
    )
    
    # Step 1: Download sequences from NCBI
    print("\nStep 1: Downloading sequences from NCBI...")
    data_processor.download_sequences()
    
    # Step 2: Process the data
    print("\nStep 2: Processing DNA sequences...")
    (X_train, y_train), (X_test, y_test) = data_processor.split_train_test()
    
    # Get input shape for the CNN
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    
    # Step 3: Optimize CNN hyperparameters using PSO
    print("\nStep 3: Optimizing CNN hyperparameters using PSO...")
    pso_optimizer = PSOOptimizer(
        n_particles=10,
        dimensions=7,
        max_iter=15
    )
    
    best_hyperparameters, best_score = pso_optimizer.optimize(X_train, y_train, input_shape)
    
    # Step 4: Train the CNN model with optimized hyperparameters
    print("\nStep 4: Training the CNN model with optimized hyperparameters...")
    cnn_model = CNNModel(input_shape, best_hyperparameters)
    
    # Create a model checkpoint
    model_path = "models/saved_models/best_model.h5"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Train the model
    history = cnn_model.train(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        model_path=model_path
    )
    
    # Step 5: Evaluate the model
    print("\nStep 5: Evaluating the model...")
    evaluator = ModelEvaluator()
    
    # Make predictions
    y_pred = cnn_model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluator.evaluate_model(cnn_model, X_test, y_test)
    
    # Plot results
    evaluator.plot_roc_curve(metrics)
    evaluator.plot_pr_curve(metrics)
    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_training_history(history)
    evaluator.visualize_predictions(X_test, y_test, y_pred)
    
    # Save metrics
    evaluator.save_metrics(metrics)
    
    print("\nExon detection project completed successfully!")
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best validation loss: {best_score:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR AUC: {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
