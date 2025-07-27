import pandas as pd
import os
import math

# File size limit in bytes (100 MB)
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

paths = [
    "data/formatted_transcripts_gzip.pkl",
    "data/formatted_transcripts_preprocessed_gzip.pkl",
]


def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except FileNotFoundError:
        return 0


def estimate_chunk_size(df, target_size_bytes):
    """
    Estimate how many rows should be in each chunk to stay under target size.

    Args:
        df: pandas DataFrame
        target_size_bytes: target size in bytes

    Returns:
        int: estimated number of rows per chunk
    """
    # Create a small sample to estimate memory usage
    sample_size = min(1000, len(df))
    sample_df = df.head(sample_size)

    # Estimate memory usage per row
    sample_memory = sample_df.memory_usage(deep=True).sum()
    memory_per_row = sample_memory / sample_size

    # Calculate rows per chunk with some safety margin (80% of target)
    rows_per_chunk = int((target_size_bytes * 0.8) / memory_per_row)

    return max(1, rows_per_chunk)  # Ensure at least 1 row per chunk


def split_dataframe_to_chunks(file_path, output_dir="data/transcripts"):
    """
    Split a large pickle file into smaller chunks.

    Args:
        file_path: path to the original pickle file
        output_dir: directory to save the chunks
    """
    print(f"Processing {file_path}...")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping...")
        return

    # Check current file size
    current_size_mb = get_file_size_mb(file_path)
    print(f"Current file size: {current_size_mb:.2f} MB")

    if current_size_mb <= MAX_FILE_SIZE_MB:
        print(f"File is already under {MAX_FILE_SIZE_MB} MB. No splitting needed.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataframe
    print("Loading dataframe...")
    df = pd.read_pickle(file_path)
    print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")

    # Estimate chunk size
    rows_per_chunk = estimate_chunk_size(df, MAX_FILE_SIZE_BYTES)
    num_chunks = math.ceil(len(df) / rows_per_chunk)

    print(f"Estimated {rows_per_chunk} rows per chunk")
    print(f"Will create approximately {num_chunks} chunks")

    # Determine output filename pattern
    base_name = os.path.basename(file_path).replace(".pkl", "")

    # Split and save chunks
    chunk_files = []
    for i in range(num_chunks):
        start_idx = i * rows_per_chunk
        end_idx = min((i + 1) * rows_per_chunk, len(df))

        chunk_df = df.iloc[start_idx:end_idx].copy()

        # Create chunk filename
        chunk_filename = f"{base_name}_chunk_{i + 1}.pkl"
        chunk_path = os.path.join(output_dir, chunk_filename)

        # Save chunk
        print(
            f"Saving chunk {i + 1}/{num_chunks}: {chunk_filename} ({len(chunk_df)} rows)"
        )
        chunk_df.to_pickle(chunk_path)

        # Verify chunk size
        chunk_size_mb = get_file_size_mb(chunk_path)
        print(f"  Chunk size: {chunk_size_mb:.2f} MB")

        chunk_files.append(chunk_path)

        if chunk_size_mb > MAX_FILE_SIZE_MB:
            print(f"  Warning: Chunk {i + 1} is still over {MAX_FILE_SIZE_MB} MB!")

    print(f"Successfully created {len(chunk_files)} chunks for {file_path}")
    return chunk_files


def main():
    """Main function to split all dataframes in paths."""
    print("Starting dataframe splitting process...")
    print(f"Target chunk size: {MAX_FILE_SIZE_MB} MB")
    print("-" * 50)

    all_chunk_files = []

    for file_path in paths:
        try:
            chunk_files = split_dataframe_to_chunks(file_path)
            if chunk_files:
                all_chunk_files.extend(chunk_files)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            print("-" * 50)
            continue

    print("Splitting process completed!")
    print(f"Total chunks created: {len(all_chunk_files)}")

    # Show directory structure
    print("\nFinal chunk directory structure:")
    try:
        for root, dirs, files in os.walk("data/transcripts"):
            level = root.replace("data/transcripts", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in sorted(files):
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    size_mb = get_file_size_mb(file_path)
                    print(f"{subindent}{file} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"Error listing directory: {e}")


if __name__ == "__main__":
    main()
