/**
 * @file main.cpp
 * @brief Demo application for the C++ audio feature extractor.
 *
 * Generates a synthetic signal (440 Hz sine wave + noise) and
 * demonstrates MFCC, log-Mel spectrogram, and energy extraction.
 */

#include "audio_features.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    // ── Configuration ───────────────────────────────────────
    audio::FeatureConfig cfg;
    cfg.sample_rate  = 16000;
    cfg.n_fft        = 512;
    cfg.hop_length   = 256;
    cfg.n_mels       = 40;
    cfg.n_mfcc       = 13;
    cfg.pre_emphasis = 0.97;

    // ── Generate synthetic signal ───────────────────────────
    int duration_samples = cfg.sample_rate * 2;  // 2 seconds
    audio::Signal signal(duration_samples);
    for (int i = 0; i < duration_samples; ++i) {
        double t = static_cast<double>(i) / cfg.sample_rate;
        signal[i] = 0.5 * std::sin(2.0 * M_PI * 440.0 * t)  // 440 Hz
                   + 0.1 * std::sin(2.0 * M_PI * 880.0 * t)  // 880 Hz harmonic
                   + 0.02 * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);  // noise
    }

    std::cout << "================================================================\n";
    std::cout << "  C++ Audio Feature Extraction Demo\n";
    std::cout << "================================================================\n\n";
    std::cout << "Signal duration : 2.0 s\n";
    std::cout << "Sample rate     : " << cfg.sample_rate << " Hz\n";
    std::cout << "FFT size        : " << cfg.n_fft << "\n";
    std::cout << "Hop length      : " << cfg.hop_length << "\n";
    std::cout << "Mel bands       : " << cfg.n_mels << "\n";
    std::cout << "MFCC coeffs     : " << cfg.n_mfcc << "\n\n";

    // ── MFCC Extraction ─────────────────────────────────────
    audio::Matrix mfcc = audio::compute_mfcc(signal, cfg);
    std::cout << "MFCC shape      : (" << mfcc.size() << ", " << cfg.n_mfcc << ")\n";

    std::cout << "\nFirst 5 frames of MFCCs:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int f = 0; f < 5 && f < static_cast<int>(mfcc.size()); ++f) {
        std::cout << "  Frame " << f << ": [";
        for (int c = 0; c < cfg.n_mfcc; ++c) {
            if (c > 0) std::cout << ", ";
            std::cout << std::setw(8) << mfcc[f][c];
        }
        std::cout << "]\n";
    }

    // ── Energy ──────────────────────────────────────────────
    audio::Matrix frames = audio::frame_signal(signal, cfg.n_fft, cfg.hop_length);
    audio::Signal energy = audio::compute_energy(frames);
    std::cout << "\nEnergy (first 10 frames):\n  [";
    for (int i = 0; i < 10 && i < static_cast<int>(energy.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::setw(8) << energy[i];
    }
    std::cout << "]\n";

    // ── Mel filterbank stats ────────────────────────────────
    audio::FilterBank fb = audio::mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate);
    std::cout << "\nMel filterbank  : (" << fb.size() << ", " << fb[0].size() << ")\n";

    // ── Log-Mel spectrogram ─────────────────────────────────
    audio::Matrix ps = audio::power_spectrum(frames, cfg.n_fft);
    audio::Matrix log_mel = audio::log_mel_spectrogram(ps, fb);
    std::cout << "Log-Mel shape   : (" << log_mel.size() << ", " << cfg.n_mels << ")\n";

    std::cout << "\n================================================================\n";
    std::cout << "  Demo complete.\n";
    std::cout << "================================================================\n";

    return 0;
}
