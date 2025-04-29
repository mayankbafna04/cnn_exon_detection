import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.data_processing import DNADataProcessor
from src.cnn_model import CNNModel
from src.pso_optimizer import PSOOptimizer
from src.evaluation import ModelEvaluator

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    accession_numbers = [ "X57436", "X80536", "X14672", "X65633", "X63128", "X66503", "Z11162",
    "Y00106", "X52473", "V01511", "X68790", "X65784", "X55039", "Z25587",
    "X70251", "X60542", "X60382", "X55545", "X55760", "X62421", "X68302",
    "X12794", "X16545", "X55741", "X80915", "X66310", "Z23091", "X57130",
    "X57127", "X60481", "X65858", "X63337", "X64994", "X64995", "X03473",
    "X76786", "X00089", "X15265", "V00532", "X55293", "X52560", "X60201",
    "Z11901", "X12458", "X73424", "V00571", "X05246", "X13556", "X83416",
    "X53065", "X79235", "Z27113", "X52259", "X52075", "X71135", "X82554",
    "X55543", "X73534", "X82676", "U01212", "U03486", "U03735", "U10116",
    "U10273", "U10360", "U10554", "U11424", "U13666", "U13695", "U16812",
    "U17894", "U18548", "U20734", "U21051", "U22346", "L10381", "D13538",
    "L19704", "M11567", "L18972", "L37019", "M35160", "M27394", "M90355",
    "M90356", "M92269", "M31423", "J00119", "L15296", "M28170", "M14333",
    "L35240", "M60119", "M90439", "M55267", "M60830", "L10820", "D16826",
    "M69199", "J04152", "M86522", "M16514", "M22403", "M80478", "L36149",
    "M60094", "M97508", "M64799", "D29685", "M22005", "M26685" ]  # as before

    data_processor = DNADataProcessor(
        accession_numbers=accession_numbers,
        email="your.email@example.com"
    )
    print("\nStep 1: Downloading sequences from NCBI...")
    data_processor.download_sequences()
    print("\nStep 2: Processing DNA sequences...")
    (X_train, y_train), (X_test, y_test) = data_processor.split_train_test()
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")

    print("\nStep 3: Optimizing CNN hyperparameters using PSO...")
    pso_optimizer = PSOOptimizer(
        n_particles=10,
        dimensions=9,
        max_iter=15
    )
    best_hyperparameters, best_score = pso_optimizer.optimize(X_train, y_train, input_shape)

    print("\nStep 4: Training the CNN model with optimized hyperparameters...")
    model_path = "models/saved_models/best_model.h5"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Initialize model
    cnn_model = CNNModel(input_shape, best_hyperparameters)
    
    # Check for existing checkpoint
    if os.path.exists(model_path):
        print("\nLoading existing checkpoint...")
        cnn_model.load(model_path)  # Load full model
        initial_epoch = len(cnn_model.model.optimizer.iterations.numpy())
    else:
        initial_epoch = 0
    
    # Configure checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,  # Save full model
        verbose=1
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Compute class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    y_train_flat = y_train.flatten()
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train_flat)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Train with checkpoint resuming
    print(f"\nResuming training from epoch {initial_epoch}...")
    history = cnn_model.model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        initial_epoch=initial_epoch,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_callback, early_stopping]
    )

    print("\nStep 5: Evaluating the model...")
    evaluator = ModelEvaluator()
    y_pred = cnn_model.predict(X_test)
    # Use lower threshold for high recall
    metrics = evaluator.evaluate_model(cnn_model, X_test, y_test, threshold=0.3)
    evaluator.plot_roc_curve(metrics)
    evaluator.plot_pr_curve(metrics)
    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_training_history(history)
    evaluator.visualize_predictions(X_test, y_test, y_pred)
    evaluator.save_metrics(metrics)

    print("\nExon detection project completed successfully!")
    print(f"Best hyperparameters: {best_hyperparameters}")
    print(f"Best validation loss: {best_score:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR AUC: {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()