import os
import logging
import threading
from pathlib import Path

# Disable Hugging Face progress bars before importing huggingface_hub.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, login

from floor_segmentation import logger


load_dotenv()


class HuggingFaceSyncer(threading.Thread):

    def __init__(
        self,
        save_dir="artifacts/model_trainer",
        interval=180,
    ):
        super().__init__(daemon=True)

        # Store the training artifact directory as an absolute path.
        self.save_dir = Path(save_dir).resolve()

        # Define the synchronization interval in seconds.
        self.interval = interval

        # Create an event used to stop the background thread.
        self.stop_event = threading.Event()

        # Read Hugging Face credentials from environment variables.
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_repo = os.getenv("HF_REPO_ID")

        # Create a directory for Hugging Face internal logs.
        self.log_dir = self.save_dir / "logs"
        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Define the Hugging Face log file.
        self.hf_log_file = self.log_dir / "huggingface.log"

        # Configure a dedicated file handler for Hugging Face logs.
        self.hf_file_handler = logging.FileHandler(
            self.hf_log_file,
            encoding="utf-8",
        )

        self.hf_file_handler.setLevel(
            logging.DEBUG
        )

        # Define the format for Hugging Face log entries.
        self.hf_file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            )
        )

        # Configure Hugging Face Hub logging.
        self._configure_huggingface_logging()

        # Configure HTTPX logging used by Hugging Face.
        self._configure_logger(
            "httpx"
        )

        # Configure HTTPCore logging used by HTTPX.
        self._configure_logger(
            "httpcore"
        )

        # Disable Hugging Face synchronization when credentials are missing.
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
                add_to_git_credential=False,
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

    def _configure_huggingface_logging(self):

        # Configure the main Hugging Face logger.
        hf_logger = logging.getLogger(
            "huggingface_hub"
        )

        # Remove existing handlers to prevent duplicate terminal output.
        hf_logger.handlers.clear()

        # Send Hugging Face logs only to the log file.
        hf_logger.addHandler(
            self.hf_file_handler
        )

        # Capture detailed Hugging Face logs.
        hf_logger.setLevel(
            logging.DEBUG
        )

        # Prevent Hugging Face logs from reaching the root logger.
        hf_logger.propagate = False

    def _configure_logger(
        self,
        logger_name,
    ):

        # Configure a third-party logger to write only to the HF log file.
        target_logger = logging.getLogger(
            logger_name
        )

        # Remove existing handlers to prevent terminal output.
        target_logger.handlers.clear()

        # Send the logger output to the HF log file.
        target_logger.addHandler(
            self.hf_file_handler
        )

        # Capture detailed logs.
        target_logger.setLevel(
            logging.DEBUG
        )

        # Prevent the logs from reaching the root logger.
        target_logger.propagate = False

    def restore_last_checkpoint(self):

        # Return immediately if Hugging Face is unavailable.
        if not self.api:

            logger.info(
                "[HF Restore] Hugging Face is not configured."
            )

            return None

        # This is the exact checkpoint path shown in the Hugging Face repository.
        hf_checkpoint_path = (
            "train/weights/last.pt"
        )

        logger.info(
            "[HF Restore] Checking for existing checkpoint..."
        )

        try:

            # Download the checkpoint into the local training artifact directory.
            downloaded_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_checkpoint_path,
                repo_type="model",
                token=self.hf_token,
                local_dir=str(
                    self.save_dir
                ),
            )

            downloaded_path = Path(
                downloaded_path
            ).resolve()

            # Verify that the downloaded checkpoint exists.
            if downloaded_path.exists():

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
                "[HF Restore] Checkpoint could not be found locally."
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
                or "does not exist" in error_text
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
                "[HF Restore] Failed to restore checkpoint."
            )

            logger.exception(e)

            raise

    def run(self):

        # Stop immediately when Hugging Face is unavailable.
        if not self.api:
            return

        # Keep synchronization running in the background.
        while not self.stop_event.is_set():

            try:

                self._sync_files()

            except Exception as e:

                # Keep synchronization errors inside the application log.
                logger.error(
                    "[HF Watcher] Synchronization failed."
                )

                logger.exception(e)

            # Wait before starting the next synchronization.
            self.stop_event.wait(
                self.interval
            )

    def _sync_files(self):

        # Return if Hugging Face is unavailable.
        if not self.api:
            return

        # Return if the training directory does not exist.
        if not self.save_dir.exists():
            return

        try:

            # Upload all training artifacts, checkpoints, and logs.
            self.api.upload_folder(
                folder_path=str(
                    self.save_dir
                ),
                repo_id=self.hf_repo,
                repo_type="model",
                commit_message="Auto-sync training artifacts",
            )

            # Keep the terminal output short and clean.
            logger.info(
                "[HF Watcher] Training artifacts synchronized."
            )

        except Exception as e:

            logger.error(
                "[HF Watcher] Training artifacts synchronization failed."
            )

            logger.exception(e)

            raise

    def stop(self):

        # Signal the background thread to stop.
        self.stop_event.set()

        # Perform one final synchronization.
        if self.api:

            try:

                self._sync_files()

                logger.info(
                    "[HF Watcher] Final synchronization completed."
                )

            except Exception as e:

                logger.error(
                    "[HF Watcher] Final synchronization failed."
                )

                logger.exception(e)

        # Close the dedicated log handler.
        self.close()

    def close(self):

        # Flush and close the Hugging Face log file.
        try:

            self.hf_file_handler.flush()
            self.hf_file_handler.close()

        except Exception:
            pass