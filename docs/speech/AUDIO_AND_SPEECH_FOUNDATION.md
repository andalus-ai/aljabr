# Aljabr Audio and Speech Foundation

## Purpose and ownership

Aljabr provides reusable Java audio data contracts and model capabilities. It does not own HTTP endpoints, serving orchestration, request scheduling, deployment, or user-facing streaming policy; those belong to AQLI and consuming applications.

The first foundation consists of:

- `:spi:aljabr-spi-audio`: stable audio format, buffer, and source contracts.
- `:spi:aljabr-spi-speech`: model-neutral ASR/STT and TTS contracts.
- `:engine:audio:aljabr-audio-core`: portable heap-backed reference audio buffer.

These modules are discovered by the existing Gradle project discovery in `settings.gradle.kts`; no explicit include list or discovery rewrite is needed.

## Audio representation and ownership

`AudioBuffer` represents interleaved, normalized floating-point PCM. One frame contains one sample per channel. The SPI exposes a read-only borrowed `FloatBuffer` view; the view must not outlive its owning AudioBuffer.

The reference `HeapAudioBuffer.copyOf` copies caller data and owns that copy. Closing the buffer ends its validity. Future native/FFM implementations must preserve the same caller-visible lifetime and ownership semantics, document whether data is borrowed or owned, and avoid silently retaining borrowed memory.

The current contract deliberately does not prescribe file codecs, resampling, channel mixing, VAD, mel extraction, tokenization, or backend placement. Those are separate reusable capabilities and should be added only after auditing the existing Whisper and EmbeddingGemma2 implementations.

## Speech contracts

- `SpeechRecognizer` accepts a `SpeechRequest` and returns a `SpeechResult`.
- `SpeechSynthesizer` accepts a `SpeechSynthesisRequest` and returns caller-owned `SpeechOutput`.
- `SpeechSegment` carries optional timing values; untimed segments use NaN for both boundaries.
- ASR and STT are names for the same recognition capability, not separate APIs.

These contracts are synchronous and model-neutral. They do not define streaming sessions, cancellation, backpressure, batching, language-specific decoding, diarization, or serving protocols. Those require dedicated lifecycle contracts and should not be smuggled into a callback-only method.

## Existing model integrations

The existing Whisper, SpeechT5, Bark, Qwen2-Audio, and EmbeddingGemma2 audio implementations remain in place. This foundation does not move or duplicate their code. A follow-up consolidation should inspect actual consumers and tests before extracting shared preprocessing or feature code.

## Dependency rules

- The audio SPI has no dependency on model implementations, codecs, tensor backends, HTTP frameworks, or native runtimes.
- The speech SPI depends only on the audio SPI.
- The reference audio core depends only on the audio SPI.
- Model modules depend inward on these contracts; the contracts never depend on model modules.
- AQLI may adapt these capabilities to its serving and streaming contracts without adding an AQLI dependency to Aljabr.

## Capability maturity

The heap audio buffer is a reference storage implementation, not a codec or audio frontend. The presence of these contracts does not imply ASR/TTS model inference is implemented or validated.
