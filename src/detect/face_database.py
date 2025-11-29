"""Face database for known faces and recognition."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from .embedder import FaceEmbedder


@dataclass
class KnownPerson:
    """A known person with their face embeddings."""
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None

    def compute_mean_embedding(self) -> None:
        """Compute mean embedding from all embeddings."""
        if self.embeddings:
            stacked = np.stack(self.embeddings)
            mean = np.mean(stacked, axis=0)
            self.mean_embedding = mean / np.linalg.norm(mean)


class FaceDatabase:
    """Database of known faces for recognition."""

    def __init__(
        self,
        embedder: FaceEmbedder,
        known_faces_dir: str = "known-faces",
        similarity_threshold: float = 0.4,
    ):
        """Initialize face database.

        Args:
            embedder: FaceEmbedder instance for computing embeddings
            known_faces_dir: Directory containing known faces
            similarity_threshold: Minimum cosine similarity for a match
        """
        self.embedder = embedder
        self.known_faces_dir = Path(known_faces_dir)
        self.similarity_threshold = similarity_threshold
        self.known_people: Dict[str, KnownPerson] = {}

    def load_known_faces(self) -> int:
        """Load all known faces from directory.

        Returns:
            Number of people loaded
        """
        self.known_people.clear()

        if not self.known_faces_dir.exists():
            print(f"Known faces directory not found: {self.known_faces_dir}")
            return 0

        # Each subdirectory is a person
        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue

            name = person_dir.name
            person = KnownPerson(name=name)

            # Load all images for this person
            image_count = 0
            for image_path in person_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                # Compute embedding
                embedding = self.embedder.get_embedding(image)
                person.embeddings.append(embedding)
                image_count += 1

            # Also check for png files
            for image_path in person_dir.glob("*.png"):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                embedding = self.embedder.get_embedding(image)
                person.embeddings.append(embedding)
                image_count += 1

            if person.embeddings:
                person.compute_mean_embedding()
                self.known_people[name] = person
                print(f"  Loaded {name}: {image_count} images")

        print(f"Loaded {len(self.known_people)} known people")
        return len(self.known_people)

    def find_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the best matching known person.

        Args:
            embedding: Face embedding to match

        Returns:
            Tuple of (name, similarity) or (None, 0.0) if no match
        """
        if not self.known_people:
            return None, 0.0

        best_name = None
        best_similarity = -1.0

        for name, person in self.known_people.items():
            if person.mean_embedding is None:
                continue

            similarity = FaceEmbedder.compute_similarity(embedding, person.mean_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        if best_similarity >= self.similarity_threshold:
            return best_name, best_similarity
        else:
            return None, best_similarity

    def get_person_count(self) -> int:
        """Get number of known people."""
        return len(self.known_people)

    def get_person_names(self) -> List[str]:
        """Get list of known person names."""
        return list(self.known_people.keys())


class UnknownFaceTracker:
    """Tracks and saves unknown faces with rate limiting."""

    def __init__(
        self,
        output_dir: str = "new-faces",
        min_save_interval: float = 1.0,
    ):
        """Initialize unknown face tracker.

        Args:
            output_dir: Base directory for saving unknown faces
            min_save_interval: Minimum seconds between saves
        """
        self.output_dir = Path(output_dir)
        self.min_save_interval = min_save_interval
        self.last_save_time = 0.0
        self.current_dir: Optional[Path] = None
        self.save_count = 0

    def _get_current_dir(self) -> Path:
        """Get or create current timestamp directory."""
        timestamp = time.strftime("%Y-%m-%d-%H-%M")
        dir_path = self.output_dir / timestamp

        if self.current_dir != dir_path:
            self.current_dir = dir_path
            self.save_count = 0

            # Count existing files if directory exists
            if dir_path.exists():
                self.save_count = len(list(dir_path.glob("*.jpg")))

        return dir_path

    def maybe_save_face(self, face_image: np.ndarray) -> bool:
        """Save face image if enough time has passed.

        Args:
            face_image: Face image to save (already cropped/scaled)

        Returns:
            True if saved, False if rate limited
        """
        current_time = time.time()

        if current_time - self.last_save_time < self.min_save_interval:
            return False

        # Get/create directory
        save_dir = self._get_current_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        self.save_count += 1
        filename = save_dir / f"{self.save_count:03d}.jpg"
        cv2.imwrite(str(filename), face_image)

        self.last_save_time = current_time
        return True
