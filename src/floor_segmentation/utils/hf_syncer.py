import os
import time
import threading
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from floor_segmentation import logger

load_dotenv()

class HuggingFaceSyncer(threading.Thread):
    def __init__(self, save_dir, interval=180):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.interval = interval
        self.stop_event = threading.Event()
        
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_repo = os.getenv("HF_REPO_ID")
        
        if not self.hf_token or not self.hf_repo:
            logger.warning("HF_TOKEN or HF_REPO_ID not set in environment! HF Sync is disabled.")
            self.api = None
        else:
            login(token=self.hf_token)
            self.api = HfApi()
            logger.info("Successfully authenticated with HuggingFace Hub.")

    def run(self):
        if not self.api:
            return

        logger.info("[HF Watcher] Background monitoring started for ENTIRE directory...")

        while not self.stop_event.is_set():
            try:
                self._sync_files()
            except Exception as e:
                logger.error(f"[HF Watcher Error]: {e}")
            
            # Non-blocking wait
            self.stop_event.wait(self.interval)

    def _sync_files(self):
        # FIX: Agar path me 'train' nahi hai ya direct model_trainer par hai, toh uske parent/base directory ko target karo
        # Ye ensure karega ki artifacts/model_trainer/ ke andar ka 'train' folder bhi sync ho jaye
        base_dir = os.path.abspath(self.save_dir)
        
        # Agar path ke andar 'train' ya specific run folder nahi hai, toh check karo ki koi subfolder hai kya
        target_path = base_dir
        if os.path.exists(base_dir):
            # Agar 'artifacts/model_trainer' pass hua hai, toh uske andar ke subfolders ko bhi pakdega
            subfolders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if subfolders:
                # Sabse latest modified subfolder ko target karenge (jaise 'train')
                latest_subfolder = max(subfolders, key=os.path.getmtime)
                if os.path.exists(os.path.join(latest_subfolder, "weights")):
                    target_path = latest_subfolder

        if os.path.exists(target_path):
            self.api.upload_folder(
                folder_path=target_path,
                repo_id=self.hf_repo,
                repo_type="model",
                commit_message="Auto-sync training artifacts and weights"
            )
            logger.info(f"[HF Watcher] All training files synced from '{target_path}' to HF repo '{self.hf_repo}'! 🚀")

    def stop(self):
        logger.info("[HF Watcher] Stopping watcher and executing final complete sync...")
        self.stop_event.set()
        if self.api:
            try:
                self._sync_files()
                logger.info("[HF Watcher] Final full sync completed successfully.")
            except Exception as e:
                logger.error(f"[HF Watcher Final Sync Error]: {e}")