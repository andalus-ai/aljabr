# Aljabr Audio and Speech Foundation

## Purpose and ownership

Aljabr provides reusable Java audio data contracts and model capabilities. It does not own HTTP endpoints, serving orchestration, request scheduling, deployment, or user-facing streaming policy; those belong to AQLI and consuming applications.

The foundation consists of:

- `:spi:aljabr-spi-audio`: stable audio format, buffer, and source contracts.
- `:spi:aljabr-spi-speech`: model-neutral ASR/STT and TTS contracts.
- `:engine:audio:aljabr-audio-core`: portable heap-backed audio storage and reusable PCM transforms.

## Audio representation and ownership

`AudioBuffer` represents interleaved, normalized floating-point PCM. One frame contains one sample per channel. The SPI exposes a read-only borrowed `FloatBuffer` view; the view must not outlive its owning AudioBuffer.

The reference `HeapAudioBuffer.copyOf` copies caller data and owns that copy. Closing the buffer ends its validity. Future native/FFM implementations must preserve the same caller-visible lifetime and ownership semantics, document whether data is borrowed or owned, and avoid silently retaining borrowed memory.

## Reusable PCM processing

The audio core provides model-neutral primitives:

- `PcmChannelMixer.downmixToMono` averages finite, clamped input channels into a new mono buffer.
- `WindowedSincPcmResampler.resample` changes sample rate independently per channel, preserving channel count and layout.

Both operations leave the input buffer caller-owned and return a new independently owned heap buffer. They are portable reference implementations; they do not claim bit-exact parity with a specific codec library or production DSP engine.

Whisper delegates channel mixing and sample-rate conversion to these shared primitives. Its 16 kHz target, 30-second windows, Mel filter bank, FFT geometry and log-Mel normalization remain Whisper-specific and stay in the Whisper model module. Do not extract model-specific feature semantics into generic audio APIs merely because another model also uses an FFT or Mel representation.

## Speech contracts

- `SpeechRecognizer` accepts a `SpeechRequest` and returns a `SpeechResult`.
- `SpeechSynthesizer` accepts a `SpeechSynthesisRequest` and returns caller-owned `SpeechOutput`.
- `SpeechSegment` carries optional timing values; untimed segments use NaN for both boundaries.
- ASR and STT are names for the same recognition capability, not separate APIs.

These contracts are synchronous and model-neutral. They do not define streaming sessions, cancellation, backpressure, batching, language-specific decoding, diarization, or serving protocols. Those require dedicated lifecycle contracts and should not be smuggled into a callback-only method.

## Existing model integrations

Existing Whisper, SpeechT5, Bark, Qwen2-Audio, and EmbeddingGemma2 implementations remain in place. This extraction is deliberately narrow: shared PCM channel mixing and resampling move into the audio engine, while Whisper's feature extraction remains model-owned. Continue auditing other model frontends before consolidating additional code.

## Dependency rules

- The audio SPI has no dependency on model implementations, codecs, tensor backends, HTTP frameworks, or native runtimes.
- The speech SPI depends only on the audio SPI.
- The reference audio core depends only on the audio SPI.
- Model modules may depend on reusable audio engine implementations; audio SPIs and engine modules must never depend on model modules.
- AQLI may adapt these capabilities to its serving and streaming contracts without adding an AQLI dependency to Aljabr.

## Capability maturity

The heap audio buffer and PCM transforms are portable reference implementations, not codecs, model frontends, or validated ASR/TTS inference. The presence of an audio frontend, checkpoint inspector, model-family descriptor or tokenizer is not evidence of executable or validated speech inference.
