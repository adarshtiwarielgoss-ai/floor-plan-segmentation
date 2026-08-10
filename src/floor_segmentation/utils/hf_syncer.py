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
        if os.path.exists(self.save_dir):
            # Poore folder ke saare content (plots, weights, logs, images) ko upload karega
            self.api.upload_folder(
                folder_path=self.save_dir,
                repo_id=self.hf_repo,
                repo_type="model",
                commit_message="Auto-sync training artifacts and weights"
            )
            logger.info(f"[HF Watcher] All training files synced to HF repo '{self.hf_repo}'! 🚀")

    def stop(self):
        logger.info("[HF Watcher] Stopping watcher and executing final complete sync...")
        self.stop_event.set()
        if self.api:
            try:
                self._sync_files()
                logger.info("[HF Watcher] Final full sync completed successfully.")
            except Exception as e:
                logger.error(f"[HF Watcher Final Sync Error]: {e}")