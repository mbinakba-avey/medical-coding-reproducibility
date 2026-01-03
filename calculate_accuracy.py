import pandas as pd
import numpy as np

def calculate_accuracy(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_feather(file_path)
    
    # Identify non-code columns
    exclude_cols = {'target', '_id'}
    code_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Found {len(code_cols)} code columns.")
    
    # Get the code with maximum probability for each row
    # idxmax returns the column name of the max value
    print("Finding top predictions...")
    top_predictions = df[code_cols].idxmax(axis=1)
    
    # Check correctness
    correct_count = 0
    total_count = len(df)
    
    print("Calculating accuracy...")
    for idx, predicted_code in top_predictions.items():
        actual_targets = df.at[idx, 'target']
        # actual_targets is a list or array of codes
        if predicted_code in actual_targets:
            correct_count += 1
            
    accuracy = correct_count / total_count
    print(f"Total samples: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    file_path = "/root/medical-coding-reproducibility/files/hkeoa1m7/predictions_test.feather"
    calculate_accuracy(file_path)

