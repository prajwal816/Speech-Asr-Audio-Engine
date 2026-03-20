/**
 * @file audio_features.cpp
 * @brief Implementation of fast audio feature extraction.
 */

#include "audio_features.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace audio {

// ═══════════════════════════════════════════════════════════════
// Pre-emphasis
// ═══════════════════════════════════════════════════════════════

Signal pre_emphasis(const Signal& signal, double coeff) {
    if (signal.empty()) return {};
    Signal out(signal.size());
    out[0] = signal[0];
    for (size_t i = 1; i < signal.size(); ++i) {
        out[i] = signal[i] - coeff * signal[i - 1];
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════
// Framing
// ═══════════════════════════════════════════════════════════════

Matrix frame_signal(const Signal& signal, int frame_length, int hop_length) {
    if (signal.empty() || frame_length <= 0 || hop_length <= 0) return {};

    int n = static_cast<int>(signal.size());
    int num_frames = 1 + (n - frame_length) / hop_length;
    if (num_frames <= 0) num_frames = 1;

    Matrix frames(num_frames, Signal(frame_length, 0.0));
    for (int f = 0; f < num_frames; ++f) {
        int start = f * hop_length;
        for (int i = 0; i < frame_length && (start + i) < n; ++i) {
            frames[f][i] = signal[start + i];
        }
    }
    return frames;
}

// ═══════════════════════════════════════════════════════════════
// Hamming window
// ═══════════════════════════════════════════════════════════════

void apply_hamming(Matrix& frames) {
    if (frames.empty()) return;
    int N = static_cast<int>(frames[0].size());
    Signal window(N);
    for (int i = 0; i < N; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (N - 1));
    }
    for (auto& frame : frames) {
        for (int i = 0; i < N; ++i) {
            frame[i] *= window[i];
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Power spectrum (real-valued DFT, brute-force for correctness)
// ═══════════════════════════════════════════════════════════════

Matrix power_spectrum(const Matrix& frames, int n_fft) {
    int n_bins = n_fft / 2 + 1;
    Matrix ps(frames.size(), Signal(n_bins, 0.0));

    for (size_t f = 0; f < frames.size(); ++f) {
        const auto& frame = frames[f];
        int N = static_cast<int>(frame.size());
        for (int k = 0; k < n_bins; ++k) {
            double re = 0.0, im = 0.0;
            for (int n = 0; n < N; ++n) {
                double angle = 2.0 * M_PI * k * n / n_fft;
                re += frame[n] * std::cos(angle);
                im -= frame[n] * std::sin(angle);
            }
            ps[f][k] = (re * re + im * im) / n_fft;
        }
    }
    return ps;
}

// ═══════════════════════════════════════════════════════════════
// Mel conversion
// ═══════════════════════════════════════════════════════════════

double hz_to_mel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

// ═══════════════════════════════════════════════════════════════
// Mel filterbank
// ═══════════════════════════════════════════════════════════════

FilterBank mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    int n_bins = n_fft / 2 + 1;
    double mel_low  = hz_to_mel(0.0);
    double mel_high = hz_to_mel(static_cast<double>(sample_rate) / 2.0);

    // n_mels + 2 equally spaced points in Mel scale
    Signal mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
    }

    // Convert to FFT bin indices
    std::vector<int> bins(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bins[i] = static_cast<int>(
            std::floor((n_fft + 1) * mel_to_hz(mel_points[i]) / sample_rate)
        );
    }

    FilterBank fb(n_mels, Signal(n_bins, 0.0));
    for (int m = 0; m < n_mels; ++m) {
        int f_left   = bins[m];
        int f_center = bins[m + 1];
        int f_right  = bins[m + 2];

        for (int k = f_left; k < f_center && k < n_bins; ++k) {
            if (f_center != f_left) {
                fb[m][k] = static_cast<double>(k - f_left) / (f_center - f_left);
            }
        }
        for (int k = f_center; k < f_right && k < n_bins; ++k) {
            if (f_right != f_center) {
                fb[m][k] = static_cast<double>(f_right - k) / (f_right - f_center);
            }
        }
    }
    return fb;
}

// ═══════════════════════════════════════════════════════════════
// Log Mel spectrogram
// ═══════════════════════════════════════════════════════════════

Matrix log_mel_spectrogram(const Matrix& power_spec, const FilterBank& fb) {
    int n_frames = static_cast<int>(power_spec.size());
    int n_mels   = static_cast<int>(fb.size());

    Matrix log_mel(n_frames, Signal(n_mels, 0.0));
    for (int f = 0; f < n_frames; ++f) {
        for (int m = 0; m < n_mels; ++m) {
            double dot = 0.0;
            int n_bins = static_cast<int>(
                std::min(power_spec[f].size(), fb[m].size())
            );
            for (int k = 0; k < n_bins; ++k) {
                dot += power_spec[f][k] * fb[m][k];
            }
            log_mel[f][m] = std::log(std::max(dot, 1e-10));
        }
    }
    return log_mel;
}

// ═══════════════════════════════════════════════════════════════
// DCT Type-II
// ═══════════════════════════════════════════════════════════════

Matrix dct(const Matrix& log_mel, int n_mfcc) {
    if (log_mel.empty()) return {};
    int n_frames = static_cast<int>(log_mel.size());
    int K        = static_cast<int>(log_mel[0].size());

    Matrix mfccs(n_frames, Signal(n_mfcc, 0.0));
    for (int f = 0; f < n_frames; ++f) {
        for (int n = 0; n < n_mfcc; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += log_mel[f][k] * std::cos(M_PI * n * (2.0 * k + 1.0) / (2.0 * K));
            }
            mfccs[f][n] = sum;
        }
    }
    return mfccs;
}

// ═══════════════════════════════════════════════════════════════
// Full MFCC pipeline
// ═══════════════════════════════════════════════════════════════

Matrix compute_mfcc(const Signal& signal, const FeatureConfig& cfg) {
    // 1. Pre-emphasis
    Signal emphasized = pre_emphasis(signal, cfg.pre_emphasis);

    // 2. Framing
    Matrix frames = frame_signal(emphasized, cfg.n_fft, cfg.hop_length);

    // 3. Windowing
    apply_hamming(frames);

    // 4. Power spectrum
    Matrix ps = power_spectrum(frames, cfg.n_fft);

    // 5. Mel filterbank
    FilterBank fb = mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate);

    // 6. Log Mel spectrogram
    Matrix log_mel = log_mel_spectrogram(ps, fb);

    // 7. DCT → MFCCs
    return dct(log_mel, cfg.n_mfcc);
}

// ═══════════════════════════════════════════════════════════════
// Energy
// ═══════════════════════════════════════════════════════════════

Signal compute_energy(const Matrix& frames) {
    Signal energy(frames.size(), 0.0);
    for (size_t f = 0; f < frames.size(); ++f) {
        double sum_sq = 0.0;
        for (double s : frames[f]) {
            sum_sq += s * s;
        }
        energy[f] = std::sqrt(sum_sq / static_cast<double>(frames[f].size()));
    }
    return energy;
}

}  // namespace audio
