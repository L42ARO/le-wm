import h5py
import numpy as np

def print_first_and_last(name, obj):
    """Checks if an object is a dataset and prints its boundaries."""
    if isinstance(obj, h5py.Dataset):
        print(f"\n📊 Dataset: {name}")
        print(f"Shape: {obj.shape} | Type: {obj.dtype}")
        print("-" * 50)
        
        # If the dataset is empty
        if obj.shape[0] == 0:
            print("[Empty Dataset]")
            return
            
        # If the dataset has 6 or fewer rows, just print the whole thing
        if obj.shape[0] <= 6:
            print("Full Data (Small Dataset):")
            print(obj[:])
        else:
            # Print first 3 rows
            print("First 3 rows:")
            print(obj[:3])
            
            print("\n... [omitting middle rows] ...\n")
            
            # Print last 3 rows
            print("Last 3 rows:")
            print(obj[-3:])
            
        print("=" * 50)

def peek_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Peeking into: {file_path}")
            f.visititems(print_first_and_last)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Check your file path.")

if __name__ == "__main__":
    file_name = "data/pusht_expert_train.h5"
    peek_hdf5(file_name)
