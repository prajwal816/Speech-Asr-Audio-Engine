/**
 * @file audio_features.h
 * @brief Fast audio feature extraction in C++.
 *
 * Implements MFCC, Mel filterbank, and energy computation using
 * raw math (no external DSP libraries).  Suitable for edge / embedded
 * deployment where Python overhead is unacceptable.
 */

#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <cstddef>
#include <vector>

namespace audio {

// ═══════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════

using Signal    = std::vector<double>;
using Matrix    = std::vector<std::vector<double>>;
using FilterBank = Matrix;

// ═══════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════

struct FeatureConfig {
    int    sample_rate   = 16000;
    int    n_fft         = 512;
    int    hop_length    = 256;
    int    n_mels        = 40;
    int    n_mfcc        = 13;
    double pre_emphasis  = 0.97;
};

// ═══════════════════════════════════════════════════════════════
// Core functions
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1].
 */
Signal pre_emphasis(const Signal& signal, double coeff = 0.97);

/**
 * @brief Frame a signal into overlapping windows.
 * @return Matrix of shape (num_frames, n_fft).
 */
Matrix frame_signal(const Signal& signal, int frame_length, int hop_length);

/**
 * @brief Apply Hamming window to each frame in-place.
 */
void apply_hamming(Matrix& frames);

/**
 * @brief Compute power spectrum from framed signal (real-valued DFT).
 * @return Matrix of shape (num_frames, n_fft/2 + 1).
 */
Matrix power_spectrum(const Matrix& frames, int n_fft);

/**
 * @brief Convert frequency in Hz to Mel scale.
 */
double hz_to_mel(double hz);

/**
 * @brief Convert Mel scale value to Hz.
 */
double mel_to_hz(double mel);

/**
 * @brief Build a triangular Mel filterbank.
 * @return FilterBank of shape (n_mels, n_fft/2 + 1).
 */
FilterBank mel_filterbank(int n_mels, int n_fft, int sample_rate);

/**
 * @brief Apply Mel filterbank to power spectrogram and take log.
 * @return Matrix of shape (num_frames, n_mels).
 */
Matrix log_mel_spectrogram(const Matrix& power_spec, const FilterBank& fb);

/**
 * @brief Compute DCT Type-II (first n_mfcc coefficients) on each frame.
 * @return Matrix of shape (num_frames, n_mfcc).
 */
Matrix dct(const Matrix& log_mel, int n_mfcc);

/**
 * @brief Full MFCC pipeline: pre-emph → frame → window → FFT → Mel → DCT.
 * @return Matrix of shape (num_frames, n_mfcc).
 */
Matrix compute_mfcc(const Signal& signal, const FeatureConfig& cfg);

/**
 * @brief Compute RMS energy per frame.
 * @return Vector of length num_frames.
 */
Signal compute_energy(const Matrix& frames);

}  // namespace audio

#endif  // AUDIO_FEATURES_H
