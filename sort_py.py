import os
import shutil

# Define the path to the folder containing the files
source_folder = '\\Dataset'

# Define the paths for the new folders
jpg_folder = os.path.join(source_folder, 'JPG_Files')
xml_folder = os.path.join(source_folder, 'XML_Files')

# Create new folders if they don't exist
os.makedirs(jpg_folder, exist_ok=True)
os.makedirs(xml_folder, exist_ok=True)

# Loop through each file in the source folder
for file_name in os.listdir(source_folder):
    # Get the file extension
    file_ext = file_name.split('.')[-1].lower()

    # Construct the full file path
    file_path = os.path.join(source_folder, file_name)

    # Check the file type and move to the respective folder
    if file_ext == 'jpg':
        shutil.move(file_path, os.path.join(jpg_folder, file_name))
    elif file_ext == 'xml':
        shutil.move(file_path, os.path.join(xml_folder, file_name))

print("Files sorted successfully!")