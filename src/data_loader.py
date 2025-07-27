import pickle
import pandas as pd
import os
import glob
from enum import Enum
from typing import Optional
from .singleton import Singleton

TRANSCRIPTS_PATH = "data/transcripts"


class TranscriptTypes(Enum):
    UNPROCESSED = "formated_transcript_"
    PREPROCESSED = "formated_transcript_preprocessed_"


class DataLoader(Singleton):
    def __init__(self, key: Optional[str] = None):
        # Only initialize once (singleton pattern)
        if not hasattr(self, "initialized"):
            self._data_cache = {}  # Cache for different data types
            self.initialized = True

        # If a key is provided, load and cache the data
        if key is not None:
            self._ensure_data_loaded(key)

    def _ensure_data_loaded(self, key: str):
        """Ensure data for the given key is loaded and cached."""
        if key not in self._data_cache:
            self._data_cache[key] = self.__load_data(key)

    def get_data(self, key: str):
        """
        Get data for the specified key. Loads if not already cached.

        Args:
            key: The data type key (use TranscriptTypes enum values)

        Returns:
            pd.DataFrame: The requested data
        """
        self._ensure_data_loaded(key)
        return self._data_cache[key]

    @property
    def data(self):
        """
        Legacy property for backwards compatibility.
        Returns the first cached dataset or None if no data is loaded.
        """
        if not self._data_cache:
            return None
        return next(iter(self._data_cache.values()))

    def clear_cache(self):
        """Clear all cached data."""
        self._data_cache.clear()

    def get_cached_keys(self):
        """Get list of currently cached data keys."""
        return list(self._data_cache.keys())

    def __load_data(self, key: str):
        """
        Load transcript data based on the provided key.
        Handles both chunked and non-chunked files.

        Args:
            key: The data type key to load

        Returns:
            pd.DataFrame: Combined dataframe from all relevant chunks
        """
        # Determine which type of transcript to load based on key
        if key == TranscriptTypes.UNPROCESSED.value:
            file_pattern = "formatted_transcripts_gzip_chunk_*.pkl"
        elif key == TranscriptTypes.PREPROCESSED.value:
            file_pattern = "formatted_transcripts_preprocessed_gzip_chunk_*.pkl"
        else:
            raise ValueError(
                f"Unknown key: {key}. Use '{TranscriptTypes.UNPROCESSED.value}' or '{TranscriptTypes.PREPROCESSED.value}'"
            )

        # Get all chunk files matching the pattern
        chunk_pattern = os.path.join(TRANSCRIPTS_PATH, file_pattern)
        chunk_files = sorted(glob.glob(chunk_pattern))

        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk files found matching pattern: {chunk_pattern}"
            )

        # Load and combine all chunks
        dataframes = []
        for i, chunk_file in enumerate(chunk_files, 1):
            try:
                df_chunk = pd.read_pickle(chunk_file)
                dataframes.append(df_chunk)

            except Exception as e:
                print(f"Error loading chunk {chunk_file}: {e}")
                continue

        if not dataframes:
            raise RuntimeError("No chunks were successfully loaded")

        # Combine all chunks into a single dataframe
        combined_df = pd.concat(dataframes).sort_index()

        return combined_df


if __name__ == "__main__":

    #### TESTING DATA LOADER ####

    # Create a single DataLoader instance (singleton)
    loader = DataLoader()

    # Load unprocessed transcripts
    unprocessed_data = loader.get_data(TranscriptTypes.UNPROCESSED.value)
    print(f"Loaded {len(unprocessed_data)} unprocessed transcripts")
    print(unprocessed_data.head())
    print(unprocessed_data.index[0])

    # Load preprocessed transcripts (same instance, different data cached)
    preprocessed_data = loader.get_data(TranscriptTypes.PREPROCESSED.value)
    print(f"Loaded {len(preprocessed_data)} preprocessed transcripts")
    print(preprocessed_data.head())

    # Test singleton behavior
    loader2 = DataLoader()
    print(f"Same instance: {loader is loader2}")  # Should be True
    print(f"Cached keys: {loader2.get_cached_keys()}")  # Should show both keys

    # Test backwards compatibility
    loader_compat = DataLoader(TranscriptTypes.UNPROCESSED.value)
    compat_data = loader_compat.data
    if compat_data is not None:
        print(f"Backwards compatibility: {len(compat_data)} transcripts loaded")
    print(f"Same instance: {loader_compat is loader}")  # Should be True
