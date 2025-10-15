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
    
    # Compute similarities
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
    print("SIMILARITY RESULTS")
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
