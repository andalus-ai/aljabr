# Qwen3-ASR audio frontend

The upstream Qwen3-ASR processor configuration specifies 16 kHz audio, 128 Mel features, a 400-point FFT, a 160-sample hop, 30-second/480,000-sample chunks, 3,000 frames, and an 8,000-sample minimum input. See the [Qwen3-ASR processor configuration](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/blob/main/processor_config.json).

## Implementation and reuse

`Qwen3AsrAudioFeatureExtractor` is a Qwen-specific contract adapter. It delegates channel mixing, resampling, FFT, Hann windowing, Mel filterbank, framing and log-Mel scaling to Aljabr's existing `WhisperAudioPreprocessor` / `WhisperLogMelSpectrogram`. The model module adds only Qwen's minimum-duration validation and output chunk contract.

This avoids maintaining a second FFT/Mel implementation. The reuse is intentional because the published Qwen processor uses Whisper-compatible feature-extractor geometry. Any future numerical deviation must be represented as a parameterized shared frontend option with reference fixtures, not a copied implementation.

## Suling boundary

Input is decoded interleaved float PCM. Suling or another provider owns encoded-media decoding and containers. This class has no Suling dependency and implements no WAV/FLAC/MP3/Opus/AAC codec. Caller-supplied PCM remains supported.

## Validation

```bash
./gradlew :models:aljabr-model-qwen3-asr:test
```

Tests cover processor geometry, input validation and defensive feature-matrix ownership. Checkpoint-level numerical equivalence against the upstream processor remains a separate conformance task.
