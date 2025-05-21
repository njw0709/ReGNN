import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
current_working_dir = os.getcwd()
TEMP_DIR = os.path.join(current_working_dir, "../temp")
# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)
