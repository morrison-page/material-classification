"""
Handles the extraction and loading of the dataset into memory.

This module provides the DatasetLoader class for extracting image datasets from ZIP archives
and organising them into two pandas DataFrames (train & test/val).
"""

import zipfile
import pandas as pd
from pathlib import Path
from typing import Tuple


class DatasetLoader:
    """Handles the extraction and loading of image datasets from ZIP archives.
    
    This class manages the entire lifecycle of dataset preparation, from extracting
    compressed archives to parsing filenames and creating structured DataFrames for
    machine learning workflows.
    
    Attributes:
        train_df (pd.DataFrame): DataFrame containing training data with columns:
            file_path, filename, split, label, index, file_size, valid
        test_df (pd.DataFrame): DataFrame containing test/validation data with the same columns
    
    Args:
        zip_path (str): Path to the ZIP archive containing the dataset
        extract_dir (str): Directory where the dataset will be extracted
        
    Raises:
        ValueError: If no training or test data is found after extraction
    """

    def __init__(self, zip_path: str, extract_dir: str) -> None:
        """Initialise the DatasetLoader and prep the dataset.
        
        Args:
            zip_path (str): Path to the ZIP archive containing the dataset
            extract_dir (str): Directory where the dataset will be extracted
        """
        self._zip_path = Path(zip_path)
        self._extract_dir = Path(extract_dir)
        self.train_df = None
        self.test_df = None
        # Populate Dataset Properties
        self._extract_dataset()
        self._create_dataframes()

    def _extract_dataset(self) -> None:
        """Extract the ZIP archive if not already extracted.
        
        Checks if the dataset folder already exists in the extraction directory.
        If not present, extracts the ZIP archive and displays status messages.
        Extraction is skipped if the dataset folder already exists.
        """
        dataset_folder = self._extract_dir / 'dataset'
        
        if not dataset_folder.exists():
            print(f"Extracting {self._zip_path.name}...")
            with zipfile.ZipFile(self._zip_path, 'r') as zip_ref:
                zip_ref.extractall(self._extract_dir)
            print(f"Dataset extracted: {self._extract_dir.resolve()}")
        else:
            print(f"Dataset already extracted: {dataset_folder.resolve()}")

    def _create_dataframes(self) -> None:
        """Create separate training and test DataFrames from extracted files.
        
        Recursively searches for all .jpg files in the dataset folder, parses their
        filenames to extract metadata (split, label, index), and creates two separate
        DataFrames for training and test data. Each DataFrame includes file paths,
        metadata, and validation flags.
        
        Raises:
            ValueError: If no training data is found in the dataset
            ValueError: If no test data is found in the dataset
        """
        dataset_folder = self._extract_dir / 'dataset'

        files = [p for p in dataset_folder.rglob('*') if p.is_file() and p.suffix.lower() == '.jpg']
        rows = []
        for file in files:
            split, label, idx = self._parse_filename(file.name)
            rows.append({
                'file_path': str(file),
                'filename': file.name,
                'split': split,
                'label': label,
                'index': idx,
                'file_size': file.stat().st_size,
                'valid': True
            })

        self.train_df = pd.DataFrame([row for row in rows if row['split'] == 'train'])
        self.test_df = pd.DataFrame([row for row in rows if row['split'] == 'test'])

        if self.train_df.empty:
            raise ValueError("No training data found")
        if self.test_df.empty:
            raise ValueError("No test data found")

    @staticmethod
    def _parse_filename(filename: str) -> Tuple[str, str, str]:
        """Parse image filename to extract metadata components.
        
        Expected filename format: {split}_{label}_{index}.jpg
        Example: train_cardboard_001.jpg
        
        Args:
            filename (str): The filename to parse (including extension)
            
        Returns:
            Tuple[str, str, str]: A tuple containing (split, label, index) where:
                - split: 'train' or 'test'
                - label: Material class name (e.g., 'cardboard', 'glass')
                - index: Numeric identifier for the image
            
        Raises:
            ValueError: If the filename doesn't match the expected three-part format
        """
        name_without_ext = filename.rsplit('.', 1)[0]
        parts = name_without_ext.split('_')

        if len(parts) != 3:
            raise ValueError(f"File: '{filename}' doesn't match expected format :(")

        return parts[0], parts[1], parts[2]