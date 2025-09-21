#!/usr/bin/env python3
"""
Classifier training module for wound classification.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from woundseg.config import Config
from woundseg.training.data import DataLoader, DataAugmentation, validate_data_structure

logger = logging.getLogger(__name__)

class ClassifierTrainer:
    """Classifier trainer for wound vs non-wound classification."""
    
    def __init__(self, 
                 data_dir: Path,
                 output_dir: Optional[Path] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 16,
                 learning_rate: float = 0.001):
        """
        Initialize classifier trainer.
        
        Args:
            data_dir: Path to training data directory
            output_dir: Path to save trained models
            image_size: Input image size
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Config.MODELS_DIR / "trained"
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_dir, image_size)
        
        # Training history
        self.history = None
        
    def build_model(self, num_classes: int = 2) -> tf.keras.Model:
        """
        Build classification model.
        
        Args:
            num_classes: Number of classes (default: 2 for wound/non-wound)
            
        Returns:
            Compiled classification model
        """
        logger.info("Building classification model...")
        
        # Use ResNet50 as base model
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        return model
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading classification data...")
        
        # Validate data structure
        validation_results = validate_data_structure(self.data_dir)
        if not validation_results['valid']:
            raise ValueError(f"Invalid data structure: {validation_results['errors']}")
        
        # Load datasets
        train_data = self.data_loader.load_classification_data('train')
        val_data = self.data_loader.load_classification_data('val')
        test_data = self.data_loader.load_classification_data('test')
        
        logger.info(f"Training samples: {len(train_data.images)}")
        logger.info(f"Validation samples: {len(val_data.images)}")
        logger.info(f"Test samples: {len(test_data.images)}")
        
        # Create TensorFlow datasets
        train_dataset = self.data_loader.create_tf_dataset(train_data, self.batch_size, shuffle=True)
        val_dataset = self.data_loader.create_tf_dataset(val_data, self.batch_size, shuffle=False)
        test_dataset = self.data_loader.create_tf_dataset(test_data, self.batch_size, shuffle=False)
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, 
              epochs: int = 50,
              use_augmentation: bool = True,
              fine_tune_epochs: int = 10,
              early_stopping_patience: int = 10,
              reduce_lr_patience: int = 5) -> Dict[str, Any]:
        """
        Train the classification model.
        
        Args:
            epochs: Number of training epochs for feature extraction
            use_augmentation: Whether to use data augmentation
            fine_tune_epochs: Number of epochs for fine-tuning
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            Training history and results
        """
        logger.info("Starting classifier training...")
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_data()
        
        # Build model
        model = self.build_model()
        
        # Add augmentation if requested
        if use_augmentation:
            augmentation = DataAugmentation.get_classification_augmentation()
            train_dataset = train_dataset.map(
                lambda x: {'images': augmentation(x['images']), 'labels': x['labels']},
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.output_dir / 'classifier_best.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ),
            tf.keras.callbacks.CSVLogger(
                filename=self.output_dir / 'training_log.csv'
            )
        ]
        
        # Phase 1: Feature extraction (frozen base model)
        logger.info(f"Phase 1: Feature extraction for {epochs} epochs...")
        self.history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (unfreeze some layers)
        if fine_tune_epochs > 0:
            logger.info(f"Phase 2: Fine-tuning for {fine_tune_epochs} epochs...")
            
            # Unfreeze some layers for fine-tuning
            model.layers[0].trainable = True
            for layer in model.layers[0].layers[:-10]:  # Unfreeze last 10 layers
                layer.trainable = True
            
            # Recompile with lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate * 0.1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training
            fine_tune_history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in self.history.history:
                self.history.history[key].extend(fine_tune_history.history[key])
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        
        # Save final model
        final_model_path = self.output_dir / 'classifier_final.keras'
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Save training summary
        summary = {
            'model_path': str(final_model_path),
            'best_model_path': str(self.output_dir / 'classifier_best.keras'),
            'test_results': dict(zip(model.metrics_names, test_results)),
            'training_history': self.history.history,
            'config': {
                'data_dir': str(self.data_dir),
                'image_size': self.image_size,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epochs_trained': len(self.history.history['loss']),
                'fine_tune_epochs': fine_tune_epochs,
                'use_augmentation': use_augmentation
            }
        }
        
        # Save summary
        import json
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Training completed successfully!")
        return summary
    
    def evaluate_model(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate trained model on test set.
        
        Args:
            model_path: Path to trained model (if None, uses best model)
            
        Returns:
            Evaluation results
        """
        if model_path is None:
            model_path = self.output_dir / 'classifier_best.keras'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Prepare test data
        _, _, test_dataset = self.prepare_data()
        
        # Evaluate
        results = model.evaluate(test_dataset, verbose=1)
        
        # Get predictions for detailed analysis
        predictions = model.predict(test_dataset)
        predicted_classes = np.argmax(predictions, axis=1)
        
        evaluation_results = {
            'model_path': str(model_path),
            'metrics': dict(zip(model.metrics_names, results)),
            'predictions_shape': predictions.shape,
            'predicted_classes': predicted_classes.tolist(),
            'config': {
                'data_dir': str(self.data_dir),
                'image_size': self.image_size,
                'batch_size': self.batch_size
            }
        }
        
        logger.info("Model evaluation completed!")
        return evaluation_results

def train_classifier_model(data_dir: Path,
                          output_dir: Optional[Path] = None,
                          epochs: int = 50,
                          batch_size: int = 16,
                          learning_rate: float = 0.001,
                          image_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    """
    Convenience function to train classifier model.
    
    Args:
        data_dir: Path to training data directory
        output_dir: Path to save trained models
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        image_size: Input image size
        
    Returns:
        Training results
    """
    trainer = ClassifierTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return trainer.train(epochs=epochs)