#!/usr/bin/env python3
"""
Compute overall similarity between MP3 audio files using multi-feature analysis.

Features used:
- MFCC (Mel-Frequency Cepstral Coefficients): Timbral texture
- Chroma: Harmonic/melodic content
- Spectral Contrast: Peaks vs valleys (distinguishes distortion/noise)
- Zero Crossing Rate: Percussiveness/noisiness
- Tempo: Beats per minute
"""

import argparse
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path
from itertools import combinations
import sys


def extract_mfcc_features(audio_path, duration=60, n_mfcc=13):
    """
    Extract rich audio features from an audio file and average over time.
    
    Combines multiple feature types for better genre differentiation:
    - MFCC: Timbral texture (13 features)
    - Chroma: Harmonic content (12 features)
    - Spectral Contrast: Peaks vs valleys in spectrum (7 features)
    - Zero Crossing Rate: Noisiness/percussiveness (1 feature)
    - Tempo: Beats per minute (1 feature)
    
    Args:
        audio_path: Path to audio file
        duration: Number of seconds to load (default 60)
        n_mfcc: Number of MFCC coefficients to compute (default 13)
    
    Returns:
        1D numpy array of concatenated features (34 dimensions total)
    """
    try:
        # Load audio file (first 60 seconds)
        y, sr = librosa.load(audio_path, duration=duration, sr=None)
        
        # 1. MFCC features (13 features)
        # Captures timbral texture - spectral envelope
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # 2. Chroma features (12 features)
        # Captures harmonic and melodic characteristics
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 3. Spectral contrast (7 features)
        # Distinguishes between harmonic and noisy sounds
        # Higher contrast = more distinct peaks (e.g., distorted guitars)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        
        # 4. Zero crossing rate (1 feature)
        # Rate at which signal changes sign
        # Higher for noisy/percussive sounds (e.g., metal vs clean vocals)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # 5. Tempo (1 feature)
        # Beats per minute
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Ensure tempo is a scalar
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        
        # Concatenate all features into single vector
        # Total: 13 + 12 + 7 + 1 + 1 = 34 features
        features = np.concatenate([
            mfcc_mean,
            chroma_mean,
            spec_contrast_mean,
            [zcr_mean],
            [tempo]
        ])
        
        return features
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}", file=sys.stderr)
        return None


def compute_similarity(features1, features2):
    """
    Compute cosine similarity between two feature vectors.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
    
    Returns:
        Similarity score between -1 and 1 (higher is more similar)
    """
    # Cosine distance is 1 - cosine similarity
    # So similarity = 1 - distance
    similarity = 1 - cosine(features1, features2)
    return similarity


def format_time(seconds):
    """
    Format seconds as min:sec.ms
    
    Args:
        seconds: Time in seconds (float)
    
    Returns:
        Formatted string like "1:23.45"
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def compute_pairwise_sequence_similarity(audio_files, window_size=3.0, hop_size=1.0, n_mfcc=13):
    """
    Compute pairwise similarity between audio files using sequence matching.
    
    For each pair of songs, slides the shorter song as a sequence through the longer
    song to find the best matching position. Returns the best similarity score for each pair.
    
    Args:
        audio_files: List of paths to audio files
        window_size: Window size in seconds
        hop_size: Step size between windows in seconds
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        List of tuples (file1, file2, best_similarity)
    """
    print(f"\nExtracting windowed features (window={window_size}s, hop={hop_size}s)...")
    
    # Extract windowed features for all files
    windowed_data = {}
    for audio_path in audio_files:
        print(f"  Processing windows: {audio_path}")
        features_list, time_stamps = extract_windowed_features(
            audio_path, window_size, hop_size, n_mfcc
        )
        if features_list:
            windowed_data[audio_path] = (np.array(features_list), time_stamps)
            print(f"    Extracted {len(features_list)} windows")
    
    print(f"\nComputing pairwise sequence similarities...")
    
    results = []
    file_list = list(windowed_data.keys())
    
    for i, file1 in enumerate(file_list):
        for j, file2 in enumerate(file_list):
            if i >= j:
                continue
            
            features1, _ = windowed_data[file1]
            features2, _ = windowed_data[file2]
            
            print(f"  Comparing {Path(file1).name} vs {Path(file2).name}...")
            
            # Determine which is shorter (will be the "query" sequence)
            if len(features1) <= len(features2):
                query_features = features1
                target_features = features2
            else:
                query_features = features2
                target_features = features1
            
            # Normalize features for cosine similarity
            query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
            target_norm = target_features / np.linalg.norm(target_features, axis=1, keepdims=True)
            
            # Compute similarity matrix using cosine similarity
            similarity_matrix = np.dot(query_norm, target_norm.T)
            
            # Find best matching position by sliding query through target
            query_len = len(query_features)
            best_similarity = -1
            
            for start_idx in range(len(target_features) - query_len + 1):
                end_idx = start_idx + query_len
                
                # Average similarity across aligned window pairs in this position
                # Window i in query should match with window (start_idx + i) in target
                aligned_similarities = []
                for i in range(query_len):
                    aligned_similarities.append(similarity_matrix[i, start_idx + i])
                
                seq_similarity = np.mean(aligned_similarities)
                
                if seq_similarity > best_similarity:
                    best_similarity = seq_similarity
            
            results.append((file1, file2, float(best_similarity)))
    
    return results


def compute_pairwise_similarities(audio_files, duration=60, n_mfcc=13):
    """
    Compute pairwise similarities between all audio files.
    
    Args:
        audio_files: List of paths to audio files
        duration: Number of seconds to load from each file
        n_mfcc: Number of MFCC coefficients to compute
    
    Returns:
        List of tuples (file1, file2, similarity_score)
    """
    # Extract features for all files
    print(f"Extracting multi-feature vectors from {len(audio_files)} files...")
    print("  (MFCC + Chroma + Spectral Contrast + ZCR + Tempo)")
    features_dict = {}
    
    for audio_path in audio_files:
        print(f"  Processing: {audio_path}")
        features = extract_mfcc_features(audio_path, duration=duration, n_mfcc=n_mfcc)
        if features is not None:
            features_dict[audio_path] = features
    
    # Compute pairwise similarities
    print(f"\nComputing pairwise similarities...")
    results = []
    
    for file1, file2 in combinations(features_dict.keys(), 2):
        similarity = compute_similarity(features_dict[file1], features_dict[file2])
        results.append((file1, file2, similarity))
    
    return results


def extract_windowed_features(audio_path, window_size=3.0, hop_size=1.0, n_mfcc=13):
    """
    Extract features from overlapping windows of audio.
    
    Args:
        audio_path: Path to audio file
        window_size: Window size in seconds (default 3.0)
        hop_size: Step size between windows in seconds (default 1.0)
        n_mfcc: Number of MFCC coefficients (default 13)
    
    Returns:
        Tuple of (features_list, time_stamps) where:
        - features_list: List of feature vectors for each window
        - time_stamps: List of (start_time, end_time) tuples for each window
    """
    try:
        # Load full audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate window and hop sizes in samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        features_list = []
        time_stamps = []
        
        # Slide window across audio
        for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            y_window = y[start_sample:end_sample]
            
            # Extract features for this window
            mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            chroma = librosa.feature.chroma_stft(y=y_window, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            spec_contrast = librosa.feature.spectral_contrast(y=y_window, sr=sr)
            spec_contrast_mean = np.mean(spec_contrast, axis=1)
            
            zcr = librosa.feature.zero_crossing_rate(y_window)
            zcr_mean = np.mean(zcr)
            
            # Concatenate features (no tempo for short windows)
            window_features = np.concatenate([
                mfcc_mean,
                chroma_mean,
                spec_contrast_mean,
                [zcr_mean]
            ])
            
            features_list.append(window_features)
            
            # Record time stamps
            start_time = start_sample / sr
            end_time = end_sample / sr
            time_stamps.append((start_time, end_time))
        
        return features_list, time_stamps
    
    except Exception as e:
        print(f"Error processing windows for {audio_path}: {e}", file=sys.stderr)
        return [], []


def find_similar_sequences(audio_files, window_size=3.0, hop_size=1.0, sequence_length=10.0, top_n=3, n_mfcc=13):
    """
    Find most similar SEQUENCES (multiple consecutive windows) that connect files.
    
    Instead of finding individual 3s windows, finds longer sequences of consecutive
    windows that are similar across files (e.g., 10 seconds of similar music).
    
    Args:
        audio_files: List of paths to audio files
        window_size: Window size in seconds (e.g., 3.0)
        hop_size: Step size between windows in seconds (e.g., 1.0)
        sequence_length: Length of sequence to find in seconds (e.g., 10.0)
        top_n: Number of top sequences to return
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        List of similar sequences, where each is a dict with:
        - 'files': dict mapping filename -> (start_time, end_time, avg_similarity)
        - 'avg_similarity': average similarity across all file pairs for this sequence
    """
    print(f"\nExtracting windowed features (window={window_size}s, hop={hop_size}s)...")
    
    # Extract windowed features for all files
    windowed_data = {}
    for audio_path in audio_files:
        print(f"  Processing windows: {audio_path}")
        features_list, time_stamps = extract_windowed_features(
            audio_path, window_size, hop_size, n_mfcc
        )
        if features_list:
            windowed_data[audio_path] = (np.array(features_list), time_stamps)
            print(f"    Extracted {len(features_list)} windows")
    
    if len(windowed_data) < 2:
        return []
    
    # Calculate how many consecutive windows to include in a sequence
    num_windows_in_sequence = max(1, int(sequence_length / hop_size))
    
    print(f"\nFinding similar sequences of {num_windows_in_sequence} consecutive windows (~{sequence_length}s)...")
    
    # Compute pairwise similarity matrices between files
    file_list = list(windowed_data.keys())
    similarity_matrices = {}
    
    for i, file1 in enumerate(file_list):
        for j, file2 in enumerate(file_list):
            if i >= j:
                continue
                
            features1, times1 = windowed_data[file1]
            features2, times2 = windowed_data[file2]
            
            print(f"  Comparing {Path(file1).name} vs {Path(file2).name}...")
            
            # Normalize features for cosine similarity
            features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
            features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            
            # Compute similarity matrix
            similarity_matrices[(file1, file2)] = np.dot(features1_norm, features2_norm.T)
    
    # Find similar sequences
    sequences = []
    
    # Check minimum number of windows across all files
    min_windows = min(len(windowed_data[f][0]) for f in file_list)
    
    # Adjust num_windows_in_sequence if needed
    if num_windows_in_sequence > min_windows:
        print(f"  Warning: Requested sequence length ({sequence_length}s = {num_windows_in_sequence} windows)")
        print(f"           exceeds shortest file ({min_windows} windows). Adjusting to {min_windows} windows.")
        num_windows_in_sequence = min_windows
    
    for anchor_file in file_list:
        features_anchor, times_anchor = windowed_data[anchor_file]
        
        # For each possible sequence start position in anchor file
        for anchor_start_idx in range(len(times_anchor) - num_windows_in_sequence + 1):
            anchor_end_idx = anchor_start_idx + num_windows_in_sequence
            
            # Get time range for this sequence
            seq_start_time = times_anchor[anchor_start_idx][0]
            seq_end_time = times_anchor[anchor_end_idx - 1][1]
            
            # For each other file, find best matching sequence
            sequence_group = {anchor_file: (seq_start_time, seq_end_time, 1.0)}
            all_similarities = []
            
            for other_file in file_list:
                if other_file == anchor_file:
                    continue
                
                features_other, times_other = windowed_data[other_file]
                
                # Get similarity matrix
                if (anchor_file, other_file) in similarity_matrices:
                    sim_matrix = similarity_matrices[(anchor_file, other_file)]
                    # For each window in anchor sequence, get similarities to all windows in other file
                    anchor_sims = sim_matrix[anchor_start_idx:anchor_end_idx, :]
                else:
                    sim_matrix = similarity_matrices[(other_file, anchor_file)]
                    anchor_sims = sim_matrix[:, anchor_start_idx:anchor_end_idx].T
                
                # Find best matching sequence in other file by sliding window
                best_seq_similarity = -1
                best_seq_start = 0
                
                for other_start_idx in range(len(times_other) - num_windows_in_sequence + 1):
                    other_end_idx = other_start_idx + num_windows_in_sequence
                    
                    # Average similarity across all window pairs in the two sequences
                    seq_similarity = np.mean(anchor_sims[:, other_start_idx:other_end_idx])
                    
                    if seq_similarity > best_seq_similarity:
                        best_seq_similarity = seq_similarity
                        best_seq_start = other_start_idx
                
                # Record best matching sequence in other file
                best_seq_end = best_seq_start + num_windows_in_sequence
                other_start_time = times_other[best_seq_start][0]
                other_end_time = times_other[best_seq_end - 1][1]
                
                sequence_group[other_file] = (other_start_time, other_end_time, float(best_seq_similarity))
                all_similarities.append(best_seq_similarity)
            
            # Calculate average similarity across all file pairs
            avg_similarity = np.mean(all_similarities) if all_similarities else 0.0
            
            sequences.append({
                'files': sequence_group,
                'avg_similarity': avg_similarity
            })
    
    # Sort by average similarity and return top N
    sequences.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    return sequences[:top_n]


def find_similar_passages(audio_files, window_size=3.0, hop_size=1.0, top_n=3, n_mfcc=13):
    """
    Find most similar passages that connect multiple audio files.
    
    For each file, finds windows that have high average similarity to 
    the best matching windows in all other files.
    
    Args:
        audio_files: List of paths to audio files
        window_size: Window size in seconds
        hop_size: Step size between windows in seconds
        top_n: Number of top "anchor passages" to return
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        List of anchor passages, where each anchor is a dict with:
        - 'files': dict mapping filename -> (start_time, end_time, best_similarity)
        - 'avg_similarity': average similarity across all file pairs
    """
    print(f"\nExtracting windowed features (window={window_size}s, hop={hop_size}s)...")
    
    # Extract windowed features for all files
    windowed_data = {}
    for audio_path in audio_files:
        print(f"  Processing windows: {audio_path}")
        features_list, time_stamps = extract_windowed_features(
            audio_path, window_size, hop_size, n_mfcc
        )
        if features_list:
            windowed_data[audio_path] = (np.array(features_list), time_stamps)
            print(f"    Extracted {len(features_list)} windows")
    
    if len(windowed_data) < 2:
        return []
    
    print(f"\nFinding anchor passages that connect all {len(windowed_data)} files...")
    
    # For each file, compute similarity matrix to all other files
    file_list = list(windowed_data.keys())
    similarity_matrices = {}
    
    for i, file1 in enumerate(file_list):
        for j, file2 in enumerate(file_list):
            if i >= j:
                continue
                
            features1, times1 = windowed_data[file1]
            features2, times2 = windowed_data[file2]
            
            print(f"  Comparing {Path(file1).name} vs {Path(file2).name}...")
            print(f"    Computing {len(features1)} x {len(features2)} = {len(features1)*len(features2):,} window pairs...")
            
            # Normalize features for cosine similarity
            features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
            features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            
            # Compute similarity matrix
            similarity_matrices[(file1, file2)] = np.dot(features1_norm, features2_norm.T)
    
    # Find anchor passages: for each window in each file, find its best matches in all other files
    anchors = []
    
    for anchor_file in file_list:
        features_anchor, times_anchor = windowed_data[anchor_file]
        
        # For each window in anchor file
        for anchor_idx, anchor_time in enumerate(times_anchor):
            # Find best matching window in each other file
            passage_group = {anchor_file: (*anchor_time, 1.0)}  # Anchor has perfect similarity to itself
            similarities = []
            
            for other_file in file_list:
                if other_file == anchor_file:
                    continue
                
                # Get similarity matrix (handle both orderings)
                if (anchor_file, other_file) in similarity_matrices:
                    sim_matrix = similarity_matrices[(anchor_file, other_file)]
                    best_idx = np.argmax(sim_matrix[anchor_idx, :])
                    best_sim = sim_matrix[anchor_idx, best_idx]
                else:
                    sim_matrix = similarity_matrices[(other_file, anchor_file)]
                    best_idx = np.argmax(sim_matrix[:, anchor_idx])
                    best_sim = sim_matrix[best_idx, anchor_idx]
                
                _, times_other = windowed_data[other_file]
                best_time = times_other[best_idx]
                
                passage_group[other_file] = (*best_time, float(best_sim))
                similarities.append(best_sim)
            
            # Calculate average similarity across all pairs
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            anchors.append({
                'files': passage_group,
                'avg_similarity': avg_similarity
            })
    
    # Sort by average similarity and return top N
    anchors.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    return anchors[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description='Compute overall similarity between MP3 audio files using multi-feature analysis'
    )
    parser.add_argument(
        'audio_files',
        nargs='+',
        help='Paths to audio files to compare'
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=60,
        help='Duration in seconds to analyze from each file (default: 60)'
    )
    parser.add_argument(
        '-n', '--n-mfcc',
        type=int,
        default=13,
        help='Number of MFCC coefficients to compute (default: 13)'
    )
    parser.add_argument(
        '--sort',
        action='store_true',
        help='Sort results by similarity score (highest first)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: average features across entire songs (fast but loses temporal structure)'
    )
    parser.add_argument(
        '--passages',
        action='store_true',
        help='Find most similar passages (individual 3s windows) between files'
    )
    parser.add_argument(
        '--sequences',
        type=float,
        default=None,
        metavar='LENGTH',
        help='Find sequences of LENGTH seconds that connect all files (e.g., --sequences 10 for 10s sequences)'
    )
    parser.add_argument(
        '-w', '--window-size',
        type=float,
        default=3.0,
        help='Window size in seconds for passage analysis (default: 3.0)'
    )
    parser.add_argument(
        '--hop-size',
        type=float,
        default=1.0,
        help='Hop size in seconds between windows (default: 1.0)'
    )
    parser.add_argument(
        '--top-passages',
        type=int,
        default=10,
        help='Number of top similar passages/sequences to show (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if len(args.audio_files) < 2:
        print("Error: Need at least 2 audio files to compare", file=sys.stderr)
        sys.exit(1)
    
    # Check that all files exist
    for audio_file in args.audio_files:
        if not Path(audio_file).exists():
            print(f"Error: File not found: {audio_file}", file=sys.stderr)
            sys.exit(1)
    
    # Mode 1: Quick mode - overall file similarity (averaged features)
    if args.quick:
        results = compute_pairwise_similarities(
            args.audio_files,
            duration=args.duration,
            n_mfcc=args.n_mfcc
        )
        
        # Sort if requested
        if args.sort:
            results.sort(key=lambda x: x[2], reverse=True)
        
        # Print results
        print("\n" + "=" * 80)
        print("OVERALL FILE SIMILARITY (averaged features - quick mode)")
        print("=" * 80)
        
        for file1, file2, similarity in results:
            # Use just filenames for cleaner output
            name1 = Path(file1).name
            name2 = Path(file2).name
            print(f"{name1} vs {name2}: {similarity:.4f}")
        
        print("=" * 80)
        print(f"Total comparisons: {len(results)}")
    
    # Mode 2: Sequences mode - find N-second sequences that connect all files
    elif args.sequences is not None:
        sequences = find_similar_sequences(
            args.audio_files,
            window_size=args.window_size,
            hop_size=args.hop_size,
            sequence_length=args.sequences,
            top_n=args.top_passages,
            n_mfcc=args.n_mfcc
        )
        
        # Print sequence results
        print("\n" + "=" * 80)
        print(f"TOP {len(sequences)} SIMILAR SEQUENCES (connecting all {len(args.audio_files)} files)")
        print(f"(Sequence length: ~{args.sequences}s, Window size: {args.window_size}s, Hop size: {args.hop_size}s)")
        print("=" * 80 + "\n")
        
        for rank, sequence in enumerate(sequences, 1):
            print(f"Sequence #{rank} (avg similarity: {sequence['avg_similarity']:.4f}):")
            
            # Sort files for consistent display order
            sorted_files = sorted(sequence['files'].items(), key=lambda x: Path(x[0]).name)
            
            for file_path, (start_time, end_time, similarity) in sorted_files:
                filename = Path(file_path).name
                start_str = format_time(start_time)
                end_str = format_time(end_time)
                duration = end_time - start_time
                print(f"  {filename:30s} [{start_str} - {end_str}] ({duration:.1f}s)  (similarity: {similarity:.4f})")
            
            print()  # Blank line between sequences
        
        print("=" * 80)
    
    # Mode 3: Find similar passages (individual 3s windows)
    elif args.passages:
        passages = find_similar_passages(
            args.audio_files,
            window_size=args.window_size,
            hop_size=args.hop_size,
            top_n=args.top_passages,
            n_mfcc=args.n_mfcc
        )
        
        # Print passage results
        print("\n" + "=" * 80)
        print(f"TOP {len(passages)} ANCHOR PASSAGES (connecting all {len(args.audio_files)} files)")
        print(f"(Window size: {args.window_size}s, Hop size: {args.hop_size}s)")
        print("=" * 80 + "\n")
        
        for rank, anchor in enumerate(passages, 1):
            print(f"Anchor #{rank} (avg similarity: {anchor['avg_similarity']:.4f}):")
            
            # Sort files for consistent display order
            sorted_files = sorted(anchor['files'].items(), key=lambda x: Path(x[0]).name)
            
            for file_path, (start_time, end_time, similarity) in sorted_files:
                filename = Path(file_path).name
                start_str = format_time(start_time)
                end_str = format_time(end_time)
                print(f"  {filename:30s} [{start_str} - {end_str}]  (similarity: {similarity:.4f})")
            
            print()  # Blank line between anchors
        
        print("=" * 80)
    
    # Mode 4 (DEFAULT): Pairwise sequence similarity
    else:
        results = compute_pairwise_sequence_similarity(
            args.audio_files,
            window_size=args.window_size,
            hop_size=args.hop_size,
            n_mfcc=args.n_mfcc
        )
        
        # Sort if requested
        if args.sort:
            results.sort(key=lambda x: x[2], reverse=True)
        
        # Print results
        print("\n" + "=" * 80)
        print("PAIRWISE SONG SIMILARITY (sequence-based)")
        print("=" * 80)
        
        for file1, file2, similarity in results:
            # Use just filenames for cleaner output
            name1 = Path(file1).name
            name2 = Path(file2).name
            print(f"{name1} vs {name2}: {similarity:.4f}")
        
        print("=" * 80)
        print(f"Total comparisons: {len(results)}")


if __name__ == "__main__":
    main()
