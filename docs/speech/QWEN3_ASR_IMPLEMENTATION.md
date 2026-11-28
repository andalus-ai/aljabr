# Qwen3-ASR implementation

## Scope and ownership

Qwen3-ASR is a distinct audio-to-text model family. Its acoustic encoder and projection are model-specific; its text backbone should reuse Aljabr's Qwen3 implementation where compatible. It is not a Qwen2-Audio implementation.

The initial `:models:aljabr-model-qwen3-asr` module adds checkpoint-family metadata, tokenizer routing metadata, and validated 1.7B architecture configuration. This is **metadata/configuration groundwork only**. It does not yet execute ASR.

## Suling boundary: do not duplicate codecs

[Suling](https://github.com/andalus-ai/suling) remains the independent Java media/codec project. Use it for the codec/container operations it implements, including WAV/FLAC/FFmpeg encoding paths and FLAC decoding APIs. Do not add WAV, FLAC, MP3, Opus, AAC, FFmpeg bindings, a second codec registry, or a competing general-purpose PCM sample-format codec to this model module or the Aljabr audio core.

The intended flow is:

1. The application uses Suling or another audio-I/O provider to decode an encoded source into PCM.
2. An optional integration adapter maps provider PCM into Aljabr's model-neutral `AudioBuffer` contract, with an explicit copy/borrow/lifetime policy.
3. Aljabr performs model-facing PCM transformations and Qwen3-ASR feature extraction.
4. Aljabr returns text/timing structures; encoded output, where needed, goes back through the selected media provider.

The adapter must remain optional and must not make Suling a dependency of Aljabr's audio SPI, audio core, or model implementation. Do not add a compile-time Suling dependency until its published artifact coordinates/repository and compatibility policy are stable. Keep the model path usable with caller-supplied PCM.

## Architecture facts captured for the 1.7B-hf configuration

- Audio frontend: 128 Mel bins, 1024-dimensional encoder, 24 layers, 16 attention heads, 4096 FFN width.
- Acoustic projection output dimension: 2048.
- Window geometry: 50-frame half-window, 800-frame inference window, 500-frame convolution chunk.
- Text backbone: Qwen3, hidden size 2048, 28 layers, 16 query heads, 8 KV heads, head dimension 128, vocabulary size 151936.
- Special tokens: audio 151676, timestamp 151705, pad 151645, EOS 151643 and 151645.

These values are represented by `Qwen3AsrConfig.qwen3Asr17B()` and validated for dimensional consistency. Treat them as a checkpoint preset, not universal defaults for every Qwen3-ASR variant.

## Implementation stages

1. **Foundation (this change):** checkpoint metadata, immutable config, tokenizer metadata, no false inference-readiness claim.
2. **Audio frontend:** model-specific log-Mel extraction and chunk/window behavior, reusing Aljabr's PCM engine for sample-rate/channel operations. Do not copy generic PCM DSP.
3. **Acoustic encoder:** convolutional subsampling, windowed transformer encoder, checkpoint key mapping and numerical fixtures.
4. **Projection and Qwen3 integration:** project audio features into the shared text embedding space and construct the audio prompt without duplicating Qwen3 decoder layers.
5. **Recognizer:** implement `SpeechRecognizer`, language/task prompting, greedy decode first, then validated decoding strategies.
6. **Checkpoint conformance:** real 0.6B/1.7B checkpoints, Indonesian and multilingual fixtures, reference transcript comparisons, memory and performance tests.

Inference maturity stays pending until checkpoint-backed end-to-end tests pass.

## Validation

```bash
./gradlew :models:aljabr-model-qwen3-asr:test
```

The test suite currently validates config invariants only. It is not an ASR accuracy or model-execution test.
