import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(CURRENT_FILE_DIR, "../temp")
# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)
