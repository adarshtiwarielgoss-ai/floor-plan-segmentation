import os
import logging
import threading
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from floor_segmentation import logger


load_dotenv()

# Disable Hugging Face progress bars globally.
# This prevents upload/download progress from interfering with YOLO terminal output.
disable_progress_bars()


class HuggingFaceSyncer(threading.Thread):

    def __init__(
        self,
        save_dir="artifacts/model_trainer",
        interval=180,
    ):
        super().__init__(daemon=True)

        self.save_dir = Path(save_dir).resolve()
        self.interval = interval

        self.stop_event = threading.Event()

        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_repo = os.getenv("HF_REPO_ID")

        self.api = None
        self.connected = False

        # Local log file for all Hugging Face operations.
        self.hf_log_file = self.save_dir / "hf_sync.log"

        self._setup_hf_logger()

        self._connect()

    # ============================================================
    # HF LOGGING
    # ============================================================

    def _setup_hf_logger(self):

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.file_logger = logging.getLogger(
            "floor_segmentation.huggingface"
        )

        self.file_logger.setLevel(logging.INFO)

        self.file_logger.propagate = False

        # Prevent duplicate handlers.
        if not self.file_logger.handlers:

            file_handler = logging.FileHandler(
                self.hf_log_file,
                encoding="utf-8",
            )

            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )

            file_handler.setFormatter(formatter)

            self.file_logger.addHandler(file_handler)

    def _hf_log(self, message, level="info"):

        if level == "error":
            self.file_logger.error(message)

        elif level == "warning":
            self.file_logger.warning(message)

        else:
            self.file_logger.info(message)

    # ============================================================
    # CONNECT TO HUGGING FACE
    # ============================================================

    def _connect(self):

        if not self.hf_token or not self.hf_repo:

            self._hf_log(
                "HF_TOKEN or HF_REPO_ID is not configured."
            )

            return

        try:

            # Suppress Hugging Face login output.
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    with redirect_stderr(devnull):

                        login(
                            token=self.hf_token,
                            add_to_git_credential=False,
                        )

                        self.api = HfApi(
                            token=self.hf_token
                        )

            self.connected = True

            self._hf_log(
                f"Successfully connected to Hugging Face repository: "
                f"{self.hf_repo}"
            )

            # This is the ONLY HF message intentionally shown
            # in the terminal.
            logger.info(
                "[HF] Hugging Face connected successfully."
            )

        except Exception as e:

            self.connected = False
            self.api = None

            self._hf_log(
                f"Connection failed: {repr(e)}",
                level="error",
            )

            logger.error(
                "[HF] Hugging Face connection failed. "
                "Check artifacts/model_trainer/hf_sync.log."
            )

    # ============================================================
    # CHECKPOINT RESTORE
    # ============================================================

    def restore_checkpoint(self):

        local_checkpoint = (
            self.save_dir
            / "train"
            / "weights"
            / "last.pt"
        )

        if not self.connected:

            self._hf_log(
                "Restore skipped because Hugging Face is not connected.",
                level="warning",
            )

            return None

        self._hf_log(
            "Checking Hugging Face for existing train/weights/last.pt..."
        )

        try:

            # Suppress all HF download output.
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    with redirect_stderr(devnull):

                        downloaded_file = hf_hub_download(
                            repo_id=self.hf_repo,
                            filename="train/weights/last.pt",
                            repo_type="model",
                            token=self.hf_token,
                        )

            local_checkpoint.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Copy downloaded checkpoint to the exact location
            # expected by the YOLO training pipeline.
            import shutil

            shutil.copy2(
                downloaded_file,
                local_checkpoint,
            )

            self._hf_log(
                f"Existing checkpoint found and downloaded successfully: "
                f"{local_checkpoint}"
            )

            return local_checkpoint

        except Exception as e:

            error_text = str(e)

            # 404 means there is no checkpoint yet.
            if (
                "404" in error_text
                or "Entry Not Found" in error_text
                or "not found" in error_text.lower()
            ):

                self._hf_log(
                    "No existing train/weights/last.pt found on Hugging Face."
                )

                return None

            self._hf_log(
                f"Checkpoint restore failed: {repr(e)}",
                level="error",
            )

            return None

    # ============================================================
    # BACKGROUND THREAD
    # ============================================================

    def run(self):

        if not self.connected:
            return

        self._hf_log(
            f"Background Hugging Face synchronization started. "
            f"Interval={self.interval} seconds."
        )

        while not self.stop_event.is_set():

            try:
                self._sync_files()

            except Exception as e:

                self._hf_log(
                    f"Background synchronization failed: {repr(e)}",
                    level="error",
                )

            self.stop_event.wait(self.interval)

    # ============================================================
    # SYNC TRAINING FILES
    # ============================================================

    def _sync_files(self):

        if not self.connected:
            return

        if not self.save_dir.exists():

            self._hf_log(
                f"Save directory does not exist: {self.save_dir}",
                level="warning",
            )

            return

        try:

            # Upload only useful training artifacts.
            #
            # This avoids repeatedly uploading unnecessary files
            # and keeps the training terminal clean.
            allowed_patterns = [
                "train/weights/last.pt",
                "train/weights/best.pt",
                "train/results.csv",
                "train/args.yaml",
                "train/results.png",
                "train/confusion_matrix.png",
                "train/confusion_matrix_normalized.png",
                "train/labels.jpg",
                "hf_sync.log",
            ]

            # Suppress all Hugging Face output.
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    with redirect_stderr(devnull):

                        self.api.upload_folder(
                            folder_path=str(self.save_dir),
                            path_in_repo="",
                            repo_id=self.hf_repo,
                            repo_type="model",
                            allow_patterns=allowed_patterns,
                            commit_message="Auto-sync training artifacts",
                            token=self.hf_token,
                        )

            self._hf_log(
                "Training artifacts synchronized successfully."
            )

        except Exception as e:

            self._hf_log(
                f"Synchronization failed: {repr(e)}",
                level="error",
            )

    # ============================================================
    # STOP THREAD AND FINAL SYNC
    # ============================================================

    def stop(self):

        self._hf_log(
            "Stopping background synchronization..."
        )

        self.stop_event.set()

        # Wait for the background thread to finish.
        if self.is_alive():

            self.join(
                timeout=30
            )

        # Perform one final synchronization.
        if self.connected:

            try:

                self._sync_files()

                self._hf_log(
                    "Final Hugging Face synchronization completed."
                )

            except Exception as e:

                self._hf_log(
                    f"Final synchronization failed: {repr(e)}",
                    level="error",
                )