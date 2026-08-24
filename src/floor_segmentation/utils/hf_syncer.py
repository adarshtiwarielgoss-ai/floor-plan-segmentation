import os
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login, HfApi, hf_hub_download

from floor_segmentation import logger


load_dotenv()


class HuggingFaceSyncer(threading.Thread):

    def __init__(self, save_dir="artifacts/model_trainer", interval=180):
        super().__init__(daemon=True)

        # Existing behavior unchanged
        self.save_dir = os.path.abspath(save_dir)
        self.interval = interval
        self.stop_event = threading.Event()

        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_repo = os.getenv("HF_REPO_ID")

        if not self.hf_token or not self.hf_repo:
            logger.warning(
                "HF_TOKEN or HF_REPO_ID not set! HF Sync is disabled."
            )
            self.api = None

        else:
            login(token=self.hf_token)
            self.api = HfApi()

            logger.info(
                "Successfully authenticated with HuggingFace Hub."
            )

    # ==========================================================
    # NEW FUNCTION
    # ==========================================================
    def restore_last_checkpoint(self):

        """
        Checks Hugging Face for:

            weights/last.pt

        If found:
            downloads it to:

            artifacts/model_trainer/weights/last.pt

        Returns:
            local checkpoint path if found/downloaded
            None if checkpoint does not exist
        """

        if not self.api:
            logger.info(
                "[HF Restore] HuggingFace is not configured. "
                "Scratch training will be used."
            )
            return None

        hf_checkpoint_path = "weights/last.pt"

        local_weights_dir = os.path.join(
            self.save_dir,
            "weights"
        )

        local_checkpoint_path = os.path.join(
            local_weights_dir,
            "last.pt"
        )

        os.makedirs(local_weights_dir, exist_ok=True)

        logger.info(
            "[HF Restore] Checking HuggingFace for "
            f"'{hf_checkpoint_path}'..."
        )

        try:

            # Check whether the file exists on HF
            self.api.hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_checkpoint_path,
                repo_type="model",
                token=self.hf_token,
                local_dir=self.save_dir,
            )

            # Because local_dir is used, the file will be placed at:
            #
            # save_dir/weights/last.pt

            if os.path.exists(local_checkpoint_path):

                logger.info(
                    "[HF Restore] Existing last.pt found on "
                    "HuggingFace."
                )

                logger.info(
                    "[HF Restore] Checkpoint downloaded to: "
                    f"{local_checkpoint_path}"
                )

                return local_checkpoint_path

            logger.warning(
                "[HF Restore] HuggingFace reported the file, "
                "but local checkpoint was not found."
            )

            return None

        except Exception as e:

            error_message = str(e).lower()

            # File does not exist on HF
            if (
                "404" in error_message
                or "not found" in error_message
                or "entry not found" in error_message
            ):

                logger.info(
                    "[HF Restore] No existing weights/last.pt "
                    "found on HuggingFace."
                )

                logger.info(
                    "[HF Restore] Training will start from scratch."
                )

                return None

            # Any other error should NOT silently start
            # a fresh training run.
            logger.error(
                "[HF Restore] Failed while downloading "
                "existing checkpoint."
            )

            logger.exception(e)

            raise

    # ==========================================================
    # EXISTING BACKGROUND SYNC
    # ==========================================================

    def run(self):

        if not self.api:
            return

        logger.info(
            "[HF Watcher] Background monitoring started on: "
            f"{self.save_dir}"
        )

        while not self.stop_event.is_set():

            try:
                self._sync_files()

            except Exception as e:

                logger.error(
                    f"[HF Watcher Error]: {e}"
                )

            self.stop_event.wait(self.interval)

    def _sync_files(self):

        if os.path.exists(self.save_dir):

            self.api.upload_folder(
                folder_path=self.save_dir,
                repo_id=self.hf_repo,
                repo_type="model",
                commit_message="Auto-sync training artifacts"
            )

            logger.info(
                f"[HF Watcher] All files inside "
                f"'{self.save_dir}' synced to HF! 🚀"
            )

    def stop(self):

        logger.info(
            "[HF Watcher] Stopping watcher and executing final sync..."
        )

        self.stop_event.set()

        if self.api:

            try:

                self._sync_files()

                logger.info(
                    "[HF Watcher] Final sync completed successfully."
                )

            except Exception as e:

                logger.error(
                    f"[HF Watcher Final Sync Error]: {e}"
                )