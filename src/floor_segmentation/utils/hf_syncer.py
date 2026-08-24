import os

# Disable Hugging Face progress bars before importing huggingface_hub.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import threading
import logging
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login, HfApi, hf_hub_download

from floor_segmentation import logger


load_dotenv()


class HuggingFaceSyncer(threading.Thread):

    def __init__(
        self,
        save_dir="artifacts/model_trainer",
        interval=180
    ):
        super().__init__(daemon=True)

        # Store the training artifact directory as an absolute path.
        self.save_dir = os.path.abspath(save_dir)

        # Define the background synchronization interval in seconds.
        self.interval = interval

        # Create an event used to stop the background thread.
        self.stop_event = threading.Event()

        # Read Hugging Face credentials from environment variables.
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_repo = os.getenv("HF_REPO_ID")

        # Create a dedicated log directory inside the training artifacts.
        self.log_dir = Path(self.save_dir) / "logs"

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Define the Hugging Face internal log file.
        self.hf_log_file = self.log_dir / "huggingface.log"

        # Create a file handler for Hugging Face internal logs.
        self.hf_file_handler = logging.FileHandler(
            self.hf_log_file,
            encoding="utf-8"
        )

        self.hf_file_handler.setLevel(
            logging.DEBUG
        )

        # Define the format used inside the Hugging Face log file.
        self.hf_file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            )
        )

        # Configure the Hugging Face Hub logger.
        self.hf_logger = logging.getLogger(
            "huggingface_hub"
        )

        self.hf_logger.handlers.clear()

        self.hf_logger.addHandler(
            self.hf_file_handler
        )

        self.hf_logger.setLevel(
            logging.DEBUG
        )

        # Prevent Hugging Face logs from reaching the terminal.
        self.hf_logger.propagate = False

        # Configure the HTTPX logger used by Hugging Face requests.
        self.httpx_logger = logging.getLogger(
            "httpx"
        )

        self.httpx_logger.handlers.clear()

        self.httpx_logger.addHandler(
            self.hf_file_handler
        )

        self.httpx_logger.setLevel(
            logging.DEBUG
        )

        # Prevent HTTPX logs from reaching the terminal.
        self.httpx_logger.propagate = False

        # Configure the HTTPCore logger used by HTTPX.
        self.httpcore_logger = logging.getLogger(
            "httpcore"
        )

        self.httpcore_logger.handlers.clear()

        self.httpcore_logger.addHandler(
            self.hf_file_handler
        )

        self.httpcore_logger.setLevel(
            logging.DEBUG
        )

        # Prevent HTTPCore logs from reaching the terminal.
        self.httpcore_logger.propagate = False

        # Disable Hugging Face synchronization if credentials are missing.
        if not self.hf_token or not self.hf_repo:

            self.api = None

            logger.warning(
                "[HF] HF_TOKEN or HF_REPO_ID is not set. "
                "Hugging Face synchronization is disabled."
            )

            return

        # Authenticate with Hugging Face.
        try:

            login(
                token=self.hf_token,
                add_to_git_credential=False
            )

            self.api = HfApi(
                token=self.hf_token
            )

            # Show only a clean connection message in the terminal.
            logger.info(
                "[HF] Hugging Face connected successfully."
            )

        except Exception as e:

            self.api = None

            logger.error(
                "[HF] Hugging Face connection failed."
            )

            logger.exception(e)

            raise

    def restore_last_checkpoint(self):

        # Return immediately if Hugging Face is not configured.
        if not self.api:

            logger.info(
                "[HF Restore] Hugging Face is not configured."
            )

            return None

        # Define the checkpoint path inside the Hugging Face repository.
        hf_checkpoint_path = "weights/last.pt"

        # Define the local weights directory.
        local_weights_dir = (
            Path(self.save_dir) / "weights"
        )

        local_weights_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.info(
            "[HF Restore] Checking for existing checkpoint..."
        )

        try:

            # Download the latest checkpoint from Hugging Face.
            downloaded_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_checkpoint_path,
                repo_type="model",
                token=self.hf_token,
                local_dir=self.save_dir,
                local_dir_use_symlinks=False,
            )

            # Verify that the checkpoint was downloaded successfully.
            if os.path.exists(downloaded_path):

                logger.info(
                    "[HF Restore] Existing last.pt found."
                )

                logger.info(
                    "[HF Restore] Checkpoint downloaded successfully."
                )

                logger.info(
                    "[HF] Training will RESUME from checkpoint."
                )

                return str(
                    downloaded_path
                )

            logger.info(
                "[HF Restore] Checkpoint was not found locally."
            )

            logger.info(
                "[HF] Training will start from SCRATCH."
            )

            return None

        except Exception as e:

            error_text = str(e).lower()

            # Treat a missing checkpoint as a normal scratch-training case.
            if (
                "404" in error_text
                or "not found" in error_text
                or "entry not found" in error_text
            ):

                logger.info(
                    "[HF Restore] No existing last.pt found."
                )

                logger.info(
                    "[HF] Training will start from SCRATCH."
                )

                return None

            # Raise unexpected Hugging Face errors.
            logger.error(
                "[HF Restore] Failed to check or download "
                "the existing checkpoint."
            )

            logger.exception(e)

            raise

    def run(self):

        # Stop the thread immediately if Hugging Face is unavailable.
        if not self.api:
            return

        logger.info(
            "[HF Watcher] Background synchronization started."
        )

        # Continue synchronization until the stop event is triggered.
        while not self.stop_event.is_set():

            try:

                self._sync_files()

            except Exception as e:

                logger.error(
                    "[HF Watcher] Synchronization failed."
                )

                logger.exception(e)

            # Wait for the configured interval before the next sync.
            self.stop_event.wait(
                self.interval
            )

    def _sync_files(self):

        # Do nothing if Hugging Face is not configured.
        if not self.api:
            return

        # Do nothing if the training artifact directory does not exist.
        if not os.path.exists(self.save_dir):
            return

        try:

            # Upload all training artifacts and logs to Hugging Face.
            self.api.upload_folder(
                folder_path=self.save_dir,
                repo_id=self.hf_repo,
                repo_type="model",
                commit_message="Auto-sync training artifacts",
            )

            # Show only a clean synchronization message.
            logger.info(
                "[HF Watcher] Training artifacts synced successfully."
            )

        except Exception as e:

            logger.error(
                "[HF Watcher] Training artifacts sync failed."
            )

            logger.exception(e)

            raise

    def stop(self):

        logger.info(
            "[HF Watcher] Stopping background synchronization."
        )

        # Signal the background thread to stop.
        self.stop_event.set()

        # Skip the final synchronization if Hugging Face is unavailable.
        if not self.api:
            return

        try:

            # Perform one final synchronization before stopping.
            self._sync_files()

            logger.info(
                "[HF Watcher] Final synchronization completed."
            )

        except Exception as e:

            logger.error(
                "[HF Watcher] Final synchronization failed."
            )

            logger.exception(e)

    def close(self):

        # Close the dedicated Hugging Face log file handler.
        try:

            if hasattr(
                self,
                "hf_file_handler"
            ):

                self.hf_file_handler.flush()
                self.hf_file_handler.close()

        except Exception:
            pass

    def __del__(self):

        # Attempt to release the log handler during object cleanup.
        try:
            self.close()
        except Exception:
            pass