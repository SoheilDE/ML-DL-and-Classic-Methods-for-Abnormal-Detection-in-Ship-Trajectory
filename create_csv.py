import os
import pandas as pd
import re

# Set the directory path where your files are located
folder_path = r'ushant_ais\data'

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith('traj_') and filename.endswith('.txt'):
        # Read each file into a DataFrame
        traj_number = re.search(r'\d+', filename).group()
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep=';', header=0)
        print(filename)
        # Add a new column with the file name
        df['traj_number'] = traj_number

        # Append the data to the combined_data DataFrame
        combined_data = combined_data.append(df, ignore_index=True)

# Save the combined data to a single CSV file
combined_data.to_csv('combined_traj_data.csv', index=False)