import pandas as pd

def preprocess_form_data(form_data, dataset_path):
    """
    Convert form data (dict) into a clean dataframe row matching the model’s expected input features.
    """
    # Load dataset to get correct column order
    df = pd.read_csv(dataset_path)
    feature_cols = [col for col in df.columns if col.lower() != 'career']

    # Initialize input row with 0s
    input_data = {col: 0 for col in feature_cols}

    # Update based on form input
    for key, val in form_data.items():
        if key in input_data:
            input_data[key] = int(val)  # '1' for checked, '0' for unchecked

    # Return a DataFrame ready for prediction
    return pd.DataFrame([input_data])
