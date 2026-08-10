import os
import sys
import shutil
import platform
import subprocess

def auto_zip_and_download(folder_path="artifacts/model_trainer", zip_name="model_trainer_results"):
    """
    Automatically zips the specified target directory and handles auto-download 
    or directory popup based on the environment (Colab, Jupyter, Local PC, or Server).
    """
    if not os.path.exists(folder_path):
        print(f"Error: Target directory '{folder_path}' not found!")
        return

    # 1. Compress the directory into a zip file
    print(f"Compressing directory '{folder_path}'...")
    zip_file = shutil.make_archive(zip_name, 'zip', folder_path)
    abs_path = os.path.abspath(zip_file)
    print(f"Archive created successfully at: {abs_path}")

    # 2. Handle environment-specific download or file opening logic
    if 'google.colab' in sys.modules:
        # Google Colab environment: Trigger browser file download
        from google.colab import files
        print("Google Colab detected: Initiating browser download...")
        files.download(zip_file)

    elif 'IPython' in sys.modules:
        # Jupyter Notebook environment: Display an interactive download link
        try:
            from IPython.display import FileLink, display
            print("Jupyter Notebook detected: Displaying download link...")
            display(FileLink(zip_file))
        except Exception:
            pass

    else:
        # Local machine: Automatically open the output folder in file explorer
        try:
            print("Local system detected: Opening directory location...")
            dir_path = os.path.dirname(abs_path)
            if platform.system() == "Windows":
                os.startfile(dir_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", dir_path])
            else:  # Linux GUI
                subprocess.run(["xdg-open", dir_path])
        except Exception:
            pass