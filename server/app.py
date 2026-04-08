import sys
import os

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from app import app
import uvicorn

def main():
    # HF Spaces requires 0.0.0.0 and port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
