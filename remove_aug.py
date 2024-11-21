import os

def delete_files_in_directory(directory: str):
    """Delete all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):  # Check if it's a file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# List of directories to clean
directories = [
    'preprocessed_dataset/train',
    'preprocessed_dataset/val',
    'preprocessed_dataset/annotations'
]

# Delete files in each directory
for dir_path in directories:
    delete_files_in_directory(dir_path)



