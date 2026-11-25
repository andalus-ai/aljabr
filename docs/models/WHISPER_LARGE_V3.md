# OpenAI Whisper Large-v3 in Aljabr

## Scope

The optional `models:aljabr-model-whisper` module adds reusable Whisper model-family metadata, Hugging Face checkpoint recognition, and strict large-v3 architecture / processor validation.

It is deliberately separate from the deprecated Safetensors REST audio adapter. It has no HTTP server dependency and can be consumed as a Java library.

## Model identity and architecture

- Hugging Face model: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- Model type: `whisper`
- Architecture: `WhisperForConditionalGeneration`
- Encoder-decoder Transformer
- Hidden size: 1280
- Encoder / decoder layers: 32 / 32
- Attention heads: 20
- Feed-forward dimension: 5120
- Audio feature bins: 128 (large-v3)
- Maximum source positions: 1500
- Maximum target positions: 448
- Vocabulary size: 51866

Audio processor expectations are validated independently from the model config:

- 16 kHz audio sampling rate
- 30-second feature-extractor chunks
- 400-point FFT
- 160-sample hop length
- 128 log-Mel feature bins

## Module usage

Add the optional model module to a Java application:

```kotlin
implementation("tech.kayys.aljabr:aljabr-model-whisper:0.1.0-SNAPSHOT")
```

The model family is discoverable through Java `ServiceLoader` using the existing
`ModelFamilyPlugin` contract. Checkpoint inspection is available through:

```java
WhisperCheckpointInspector.isWhisperCheckpoint(modelDirectory);
WhisperCheckpointInspector.inspectLargeV3(modelDirectory);
WhisperLargeV3Config.fromCheckpoint(modelDirectory);
```

`inspectLargeV3` checks the model config, processor config, tokenizer files,
and single-file or sharded Safetensors weight index. A successful report means
the expected checkpoint files and metadata are present; it does not mean a
runtime provider can execute the model.

## Capability maturity

| Capability | State |
|---|---|
| Whisper family discovery | Implemented |
| Audio model-family declaration | Implemented |
| Hugging Face tokenizer artifact discovery | Declared through existing tokenizer SPI |
| Large-v3 architecture validation | Implemented |
| Audio processor metadata validation | Implemented |
| Safetensors checkpoint presence validation | Implemented |
| Audio resampling / log-Mel feature extraction | Not implemented in this module |
| Whisper encoder / decoder execution | Not implemented |
| Transcription / translation generation | Not implemented |
| Timestamp alignment / segment decoding | Not implemented |
| CPU / accelerator provider conformance | Not implemented |

The plugin therefore does **not** advertise `DIRECT_SAFETENSOR_INFERENCE`.
Do not interpret model-family recognition or a complete checkpoint report as
validated speech inference.

## Next implementation gate

Implement the reusable audio front-end (resampling, mono conversion, large-v3
128-bin log-Mel extraction and 30-second windowing), then the Whisper
encoder-decoder execution path against Aljabr's existing tensor and provider
contracts. Add deterministic numerical fixtures and transcription/translation
conformance tests before changing the family descriptor to claim executable
inference support.
