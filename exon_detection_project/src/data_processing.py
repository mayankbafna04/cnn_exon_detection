# src/data_processing.py
import os
import random
import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import requests
import time

class DNADataProcessor:
    def __init__(self, accession_numbers, email, raw_dir='data/raw', 
                 processed_dir='data/processed', train_dir='data/train', 
                 test_dir='data/test', train_ratio=0.8):
        """
        Initialize the DNA data processor.
        
        Parameters:
        -----------
        accession_numbers : list
            List of NCBI accession numbers to download
        email : str
            Email for NCBI Entrez queries
        raw_dir : str
            Directory to store raw sequences
        processed_dir : str
            Directory to store processed sequences
        train_dir : str
            Directory to store training data
        test_dir : str
            Directory to store testing data
        train_ratio : float
            Ratio of data to use for training (default: 0.8)
        """
        self.accession_numbers = accession_numbers
        self.email = email
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_ratio = train_ratio
        
        # Create directories if they don't exist
        for directory in [raw_dir, processed_dir, train_dir, test_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up Entrez email
        Entrez.email = email
    
    def download_sequences(self):
        """Download DNA sequences from NCBI using the accession numbers"""
        print("Downloading sequences from NCBI...")
        
        for acc in tqdm(self.accession_numbers):
            output_file = os.path.join(self.raw_dir, f"{acc}.gb")
            
            # Skip if file already exists
            if os.path.exists(output_file):
                continue
            
            try:
                # Fetch the GenBank record
                handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text")
                record = handle.read()
                
                # Save the record
                with open(output_file, 'w') as f:
                    f.write(record)
                
                handle.close()
                time.sleep(1)  # Be nice to NCBI servers
                
            except Exception as e:
                print(f"Error downloading {acc}: {e}")
    
    def extract_exons(self):
        """Extract exons from GenBank files"""
        print("Extracting exons from GenBank files...")
        
        all_sequences = []
        all_labels = []
        
        for acc in tqdm(self.accession_numbers):
            gb_file = os.path.join(self.raw_dir, f"{acc}.gb")
            
            if not os.path.exists(gb_file):
                print(f"Warning: File {gb_file} not found. Skipping.")
                continue
            
            try:
                # Parse the GenBank record
                record = SeqIO.read(gb_file, "genbank")
                sequence = str(record.seq)
                
                # Create a label array (0 for non-exon, 1 for exon)
                labels = np.zeros(len(sequence))
                
                # Mark exon regions
                for feature in record.features:
                    if feature.type == "exon":
                        start = feature.location.start
                        end = feature.location.end
                        labels[start:end] = 1
                
                all_sequences.append(sequence)
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Error processing {acc}: {e}")
        
        return all_sequences, all_labels
    
    def process_sequences(self, window_size=1000, stride=200):
        """
        Process sequences into overlapping windows with labels
        
        Parameters:
        -----------
        window_size : int
            Size of the sliding window
        stride : int
            Step size for the sliding window
        """
        print("Processing sequences into windows...")
        
        sequences, labels = self.extract_exons()
        windows = []
        window_labels = []
        
        for seq, label in zip(sequences, labels):
            seq_length = len(seq)
            
            for i in range(0, seq_length - window_size + 1, stride):
                window = seq[i:i+window_size]
                window_label = label[i:i+window_size]
                
                # Skip windows with ambiguous bases (N)
                if 'N' in window:
                    continue
                
                windows.append(window)
                window_labels.append(window_label)
        
        # Convert to one-hot encoding
        X = self.one_hot_encode(windows)
        y = np.array(window_labels)
        
        # Save processed data
        np.save(os.path.join(self.processed_dir, "X_data.npy"), X)
        np.save(os.path.join(self.processed_dir, "y_data.npy"), y)
        
        return X, y
    
    def one_hot_encode(self, sequences):
        """
        One-hot encode DNA sequences
        
        Parameters:
        -----------
        sequences : list
            List of DNA sequences
        
        Returns:
        --------
        np.ndarray
            One-hot encoded sequences
        """
        # Define mapping: A->0, C->1, G->2, T->3
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Initialize the encoded array
        X = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            for j, nucleotide in enumerate(seq):
                if nucleotide in mapping:
                    X[i, j, mapping[nucleotide]] = 1.0
        
        return X
    
    def split_train_test(self):
        """Split data into training and testing sets"""
        print("Splitting data into train and test sets...")
        
        try:
            X = np.load(os.path.join(self.processed_dir, "X_data.npy"))
            y = np.load(os.path.join(self.processed_dir, "y_data.npy"))
        except:
            X, y = self.process_sequences()
        
        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Split into train and test
        split_idx = int(self.train_ratio * X.shape[0])
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Save train and test sets
        np.save(os.path.join(self.train_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.train_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.test_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.test_dir, "y_test.npy"), y_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return (X_train, y_train), (X_test, y_test)
