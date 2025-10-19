import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def test_model_training(tmp_path):
    """
    Unit test to verify that the model trains successfully
    and achieves a reasonable accuracy on the dataset.
    """

    # Make sure dataset exists
    data_path = f"data/v2/data.csv"
    assert os.path.exists(data_path), f"Data file not found: {data_path}"

    # Run the script to train model
    os.system("python script.py")

    # Check model file creation
    model_file = f"model_v2.joblib"
    assert os.path.exists(model_file), "Model file was not created!"

    # Load model and data to verify performance
    model = joblib.load(model_file)
    data = pd.read_csv(data_path)

    X = data[['sepal_length','sepal_width','petal_length','petal_width']]
    y = data['species']

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Check model accuracy threshold
    assert acc > 0.7, f"Model accuracy too low: {acc:.3f}"
