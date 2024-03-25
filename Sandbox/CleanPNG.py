import os

folder_path = "./" 

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Iterate over the files and delete the .png files
for file in files:
    if file.endswith(".png"):
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)