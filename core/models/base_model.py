"""
Basic keras model
"""
import keras
import numpy as np
import pandas as pd
import datetime
import os


class BaseModel():
    """
    The basic model object.
    #TODO: write doc.
    """

    def __init__(self) -> None:
        raise NotImplementedError("Must NOT call BaseModel directly.")

    def save_model(
            self,
            file_dir: str=None
    ) -> None:
        if file_dir is None:
            # If no file directory specified, use current time.
            now = datetime.datetime.now()
            file_dir = now.strftime("%Y_%m_%d_%H_%M_%S")

        # Try to create folder for current session record.
        try:
            folder = f"./saved_models/{file_dir}/"
            os.system(f"mkdir {folder}")
            print(f"Session record directory created: {folder}")
        except:
            print("Current directory: ")
            _ = os.system("pwd")
            raise FileNotFoundError(
                "Failed to create directory, please create directory ./saved_models/")

        # Save model structure to JSON
        print("Saving model structure...")
        model_json = self.core.to_json()
        with open(f"{folder}model_structure.json", "w") as json_file:
            json_file.write(model_json)
        print("Done.")

        # Save model weight to h5
        print("Saving model weights...")
        self.core.save_weights(f"{folder}model_weights.h5")
        print("Done")

        # Save model illustration to png file.
        print("Saving model visualization...")
        try:
            keras.utils.plot_model(
                self.core,
                to_file=f"{folder}model.png",
                show_shapes=True,
                show_layer_names=True)
        except:
            print("Model illustration cannot be saved.")

        # Save training history (if any)
        if self.hist is not None:
            hist_loss = np.squeeze(np.array(self.hist.history["loss"]))
            hist_val_loss = np.squeeze(np.array(self.hist.history["val_loss"]))
            combined = np.stack([hist_loss, hist_val_loss])
            combined = np.transpose(combined)
            df = pd.DataFrame(combined, dtype=np.float32)
            df.columns = ["loss", "val_loss"]
            df.to_csv(f"{folder}hist.csv", sep=",")
            print(f"Training history is saved to {folder}hist.csv...")
        else:
            print("No training history found.")

        print("Done.")

    def load_model(
            self,
            folder_dir: str
    ) -> None:
        """
        #TODO: doc
        """
        if not folder_dir.endswith("/"):
            # Assert the correct format, folder_dir should be
            folder_dir += "/"

        print(f"Load model from folder {folder_dir}")

        # construct model from json
        print("Reconstruct model from Json file...")
        try:
            json_file = open(f"{folder_dir}model_structure.json", "r")
        except FileNotFoundError:
            raise Warning(
                f"Json file not found. Expected: {folder_dir}model_structure.json"
            )

        model_file = json_file.read()
        json_file.close()
        self.core = keras.models.model_from_json(model_file)
        print("Done.")

        # load weights from h5
        print("Loading model weights...")
        try:
            self.core.load_weights(
                f"{folder_dir}model_weights.h5", by_name=True)
        except FileNotFoundError:
            raise Warning(
                f"h5 file not found. Expected: {folder_dir}model_weights.h5"
            )
        print("Done.")
        self.core.compile(loss="mse", optimizer="adam")
