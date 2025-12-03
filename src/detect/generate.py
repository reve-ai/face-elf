
import requests
import base64
import json
import cv2
import numpy as np
import os
import configparser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable

ELF_PROMPT = """The person from <img>0</img> is an elf at a very glamorous Christmas party.
There are festive decorations, lights, and people in the background."""

def _load_api_key() -> str:
    """Load REVE API key from environment variable or config file.

    Checks in order:
    1. REVE_API_KEY environment variable
    2. ~/.config/detect.ini [reve] api_key

    Raises RuntimeError if not found.
    """
    # Check environment variable first
    api_key = os.environ.get('REVE_API_KEY')
    if api_key:
        return api_key

    # Check config file
    config_path = Path.home() / '.config' / 'detect.ini'
    if config_path.exists():
        config = configparser.ConfigParser()
        config.read(config_path)
        if config.has_option('reve', 'api_key'):
            api_key = config.get('reve', 'api_key')
            if api_key:
                return api_key

    raise RuntimeError(
        "REVE API key not found. Set REVE_API_KEY environment variable "
        "or add [reve] api_key=... to ~/.config/detect.ini"
    )


# API key loaded lazily on first use
_REVE_API_KEY: Optional[str] = None


def _get_api_key() -> str:
    """Get the API key, loading it on first call."""
    global _REVE_API_KEY
    if _REVE_API_KEY is None:
        _REVE_API_KEY = _load_api_key()
    return _REVE_API_KEY


def base64_encode_image(image: np.ndarray) -> str:
    """Encode a numpy image array to base64 string (via PNG encoding)."""
    success, png_data = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(png_data.tobytes()).decode('utf-8')


def base64_decode_image(b64_string: str) -> np.ndarray:
    """Decode a base64 string to a numpy image array."""
    image_bytes = base64.b64decode(b64_string)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from base64 data")
    return image

def _get_headers() -> dict:
    """Get request headers with API key."""
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

def generate_elf(image: np.ndarray) -> np.ndarray | None:
    """Generate an elf version of the input image using the Reve API."""
    image_base64 = base64_encode_image(image)
    # Set up request payload
    payload = {
        "edit_instruction": ELF_PROMPT,
        "reference_image": image_base64,
        "aspect_ratio": "16:9",
        "version": "latest-fast",
        "test_time_scaling": 3
    }

    # Make the API request
    try:
        print(f"Making an elf out of image...")
        response = requests.post("https://api.reve.com/v1/image/edit", headers=_get_headers(), json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Parse the response
        result = response.json()
        print(f"Request ID: {result['request_id']}")
        print(f"Credits used: {result['credits_used']}")
        print(f"Credits remaining: {result['credits_remaining']}")

        if result.get('content_violation'):
            print("Warning: Content policy violation detected")
        else:
            print("Image edited successfully!")
        return base64_decode_image(result['image'])
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        return None


# Thread pool for async generation
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="elf_gen")
    return _executor


def generate_elf_async(image: np.ndarray) -> Future:
    """Submit elf generation to run asynchronously.

    Returns a Future that will contain the result (np.ndarray or None).
    Check future.done() to see if complete, then future.result() to get the image.
    """
    executor = _get_executor()
    return executor.submit(generate_elf, image)


def shutdown_executor():
    """Shutdown the thread pool executor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=True)
        _executor = None
