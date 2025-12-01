"""Configuration file handling for face detection application."""

import configparser
import os
from pathlib import Path
from typing import Tuple, Optional


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "detect.ini"


class AppConfig:
    """Application configuration management."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Path to config file. Uses default if None.
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = configparser.ConfigParser()

        # Default values
        self._camera_width = 640
        self._camera_height = 480
        self._conf_threshold = 0.5
        self._similarity_threshold = 0.4

    @property
    def camera_resolution(self) -> Tuple[int, int]:
        """Get camera resolution setting."""
        return (self._camera_width, self._camera_height)

    @camera_resolution.setter
    def camera_resolution(self, value: Tuple[int, int]):
        """Set camera resolution setting."""
        self._camera_width, self._camera_height = value

    @property
    def camera_width(self) -> int:
        return self._camera_width

    @property
    def camera_height(self) -> int:
        return self._camera_height

    @property
    def conf_threshold(self) -> float:
        return self._conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, value: float):
        self._conf_threshold = value

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float):
        self._similarity_threshold = value

    def load(self) -> bool:
        """Load configuration from file.

        Returns:
            True if file was loaded, False if using defaults
        """
        if not self.config_path.exists():
            return False

        try:
            self.config.read(self.config_path)

            if "camera" in self.config:
                self._camera_width = self.config.getint("camera", "width", fallback=640)
                self._camera_height = self.config.getint("camera", "height", fallback=480)

            if "detection" in self.config:
                self._conf_threshold = self.config.getfloat(
                    "detection", "conf_threshold", fallback=0.5
                )
                self._similarity_threshold = self.config.getfloat(
                    "detection", "similarity_threshold", fallback=0.4
                )

            print(f"Loaded configuration from {self.config_path}")
            return True

        except Exception as e:
            print(f"Error loading config: {e}")
            return False

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if file was saved successfully
        """
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Update config object
            if "camera" not in self.config:
                self.config["camera"] = {}
            self.config["camera"]["width"] = str(self._camera_width)
            self.config["camera"]["height"] = str(self._camera_height)

            if "detection" not in self.config:
                self.config["detection"] = {}
            self.config["detection"]["conf_threshold"] = str(self._conf_threshold)
            self.config["detection"]["similarity_threshold"] = str(self._similarity_threshold)

            # Write to file
            with open(self.config_path, "w") as f:
                self.config.write(f)

            print(f"Saved configuration to {self.config_path}")
            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False
