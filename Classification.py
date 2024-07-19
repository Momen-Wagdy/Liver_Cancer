import os
import shutil
import pandas as pd

# Path to the directory containing the photos and CSV file
source_dir = r'D:\Projects\Omnia\Ultrasound\liver_ultrasound.v11i.tensorflow\Pre_Processed_data\test'
# Path to the directory where photos will be saved in class-based folders
destination_dir = r'D:\Projects\Omnia\Ultrasound\liver_ultrasound.v11i.tensorflow\Classified_data\test'

# Read the CSV file
csv_file_path = os.path.join(source_dir, '_annotations.csv')
data = pd.read_csv(csv_file_path)

# Iterate over each row in the CSV file
for index, row in data.iterrows():
    filename = row['filename']
    class_label = row['class']
    
    # Create the class folder if it doesn't exist
    class_folder_path = os.path.join(destination_dir, class_label)
    if not os.path.exists(class_folder_path):
        os.makedirs(class_folder_path)
    
    # Check if the source file exists
    source_file_path = os.path.join(source_dir, filename)
    if not os.path.exists(source_file_path):
        print(f"File not found: {source_file_path}")
        continue  # Skip this file and move to the next one
    
    # Move the photo to the class folder
    destination_file_path = os.path.join(class_folder_path, filename)
    shutil.move(source_file_path, destination_file_path)
    print(f"Moved {source_file_path} to {destination_file_path}")

print("Photos have been organized into class-based folders.")
