# OpenAI Whisper Large-v3 in Aljabr

## Scope

The optional `models:aljabr-model-whisper` module adds reusable Whisper model-family metadata, Hugging Face checkpoint recognition, strict large-v3 architecture / processor validation, and a pure-Java audio frontend.

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

## Audio frontend

`WhisperAudioPreprocessor` accepts interleaved float PCM, validates and downmixes channels, resamples to 16 kHz using a windowed-sinc low-pass interpolator, and splits long recordings into 30-second windows.

`WhisperLogMelSpectrogram` is the pure-Java reference frontend. It uses a 400-sample periodic Hann window, 160-sample hop, 128 Slaney-normalized Mel filters, an exact 400-point Bluestein FFT, Whisper log scaling, and returns a `[melBin][frame]` matrix with 3000 frames per window.

The processor metadata expected by the official checkpoint is validated separately:

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
`ModelFamilyPlugin` contract. Audio preparation is available through:

```java
WhisperAudioPreprocessor preprocessor = new WhisperAudioPreprocessor();
List<WhisperAudioPreprocessor.AudioChunk> chunks =
    preprocessor.prepare(interleavedPcm, sampleRate, channels);
```

Checkpoint inspection is available through:

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
| PCM channel downmixing | Implemented |
| 16 kHz windowed-sinc resampling | Implemented |
| 128-bin log-Mel extraction | Implemented as pure-Java reference |
| 30-second chunking | Implemented |
| Safetensors checkpoint presence validation | Implemented |
| Whisper tokenizer execution / task-token decoding | Not implemented |
| Whisper encoder / decoder execution | Not implemented |
| Transcription / translation generation | Not implemented |
| Timestamp alignment / segment decoding | Not implemented |
| CPU / accelerator provider conformance | Not implemented |

The plugin therefore does **not** advertise `DIRECT_SAFETENSOR_INFERENCE`.
Do not interpret model-family recognition, frontend output, or a complete
checkpoint report as validated speech inference.

## Next implementation gate

Implement the Whisper encoder-decoder execution path against Aljabr's existing
tensor and provider contracts. Integrate the tokenizer's special task tokens,
language selection, timestamp tokens, and long-form decoding state. Add
reference numerical fixtures and transcription/translation conformance tests
before changing the family descriptor to claim executable inference support.
