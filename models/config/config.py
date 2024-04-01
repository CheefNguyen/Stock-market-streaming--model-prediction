import pathlib
import pandas as pd
import datetime
import os

# DATA
TRAINING_DATA_FILE = "models/data/PastData_FullFloorHSX.csv"

now = datetime.datetime.now().strftime("%Y-%m-%d")
TRAINED_MODEL_DIR = f"models/trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

# TESTING_DATA_FILE = "test.csv"