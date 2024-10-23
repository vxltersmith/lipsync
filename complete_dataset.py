import os
import shutil

# Define directories
preprocessed_dir = "/data/CelebVHQ/35666/"
# mp4_dir = "/data/CelebV-HQ/35666/"

# # List all .mp4 files in the mp4_dir
# mp4_files = [f for f in os.listdir(mp4_dir) if f.endswith('.mp4')]

# # Iterate over the .mp4 files
# for mp4_file in mp4_files:
#     # Strip the '.mp4' extension from the file name to find corresponding pre-processed directory
#     subdir_name = mp4_file[:-4]  # Remove the '.mp4' part
    
#     # Construct paths
#     src_file = os.path.join(mp4_dir, mp4_file)
#     target_dir = os.path.join(preprocessed_dir, subdir_name)
#     target_file = os.path.join(target_dir, mp4_file)

#     # Check if the corresponding subdir exists
#     if os.path.exists(target_dir):
#         # Copy the .mp4 file to the target preprocessed subdir
#         shutil.copy(src_file, target_file)
#         print(f"Copied {mp4_file} to {target_dir}")
#     else:
#         print(f"Directory {target_dir} does not exist for {mp4_file}")

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function to get the size of an individual file
def get_file_size(file_path):
    return os.path.getsize(file_path)

# Function to calculate directory size using multiple workers
def get_directory_size(directory, max_workers=8):
    total_size = 0
    file_paths = []
    
    # Collect all file paths from the directory
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            file_paths.append(fp)
    
    # Use ThreadPoolExecutor to get file sizes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_file_size, fp): fp for fp in file_paths}
        
        # Initialize tqdm progress bar
        with tqdm(total=len(file_paths), desc="Calculating size", unit="files") as pbar:
            for future in as_completed(futures):
                total_size += future.result()  # Accumulate file size
                pbar.update(1)  # Update progress bar

    return total_size

# Step 1: Calculate the size in bytes using parallel processing and progress bar
size_in_bytes = get_directory_size(preprocessed_dir)

# Convert to Gigabytes (GB)
size_in_gb = size_in_bytes / (1024 ** 3)

print(f"Total size of {preprocessed_dir}: {size_in_gb:.2f} GB")

import tarfile

# Define the name for the output tar.gz file
output_filename = "/data/CelebVHQ/35666_preprocessed.tar.gz"

# Create a tar.gz archive
with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(preprocessed_dir, arcname=os.path.basename(preprocessed_dir))

print(f"Compressed folder into: {output_filename}")