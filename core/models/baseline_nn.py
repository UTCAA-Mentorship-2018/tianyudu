"""
Basic Neural Net
"""
import keras
import pandas as pd
import numpy as np
from core.models.base_model import BaseModel

class BaselineNN(BaseModel):
    def __init__(
        self,
        input_dim: int
    ) -> None:
        self.hist = None
        self.num_fea = input_dim
        self.core = self.build_model(self.num_fea)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int=10
    ) -> None:
        assert len(X_train) == len(y_train),\
        "Length of training set predictors and training set response must agree."
        assert len(X_val) == len(y_val),\
        "Length of validation set predictors and validation set response must agree."
        assert X_train.shape[1] == X_val.shape[1] == self.num_fea,\
        f"Number of features(columns) in training and validation set should be   "

        self.hist = self.core.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val)
        )

    def build_model(
        self,
        input_dim: int
    ) -> keras.Sequential:
        # create model
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            units=64,
            input_dim=input_dim,
            kernel_initializer='normal',
            activation='relu')
        )
        
        model.add(keras.layers.Dense(
            units=32,
            kernel_initializer="normal",
            activation="sigmoid"
        ))

        model.add(keras.layers.Dense(
            units=1,
            kernel_initializer='normal',
            activation='sigmoid')
        )
        # Compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        return model
