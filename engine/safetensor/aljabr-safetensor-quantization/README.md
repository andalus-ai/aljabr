# Aljabr SafeTensor Quantization Module

## Overview

The `aljabr-safetensor-quantization` module provides comprehensive model quantization support for SafeTensor models in the Aljabr inference engine. It enables model compression with minimal quality loss, supporting multiple quantization strategies optimized for different hardware and use cases.

## Features

### Supported Quantization Strategies

| Strategy | Bits | Compression | Best For | Quality Loss |
|----------|------|-------------|----------|--------------|
| **INT4** | 4-bit | ~8x | CPU inference, maximum compression | Low-Medium |
| **INT8** | 8-bit | ~4x | Balanced CPU/GPU inference | Very Low |
| **FP8**  | 8-bit | ~4x | GPU with FP8 tensor cores (H100, MI300) | Minimal |

### Key Capabilities

- **GPTQ (Generative Pretrained Transformer Quantization)**: Advanced 4-bit quantization using approximate second-order information
- **Per-channel quantization**: Better accuracy with channel-wise scale factors
- **Group-wise quantization**: Efficient quantization with configurable group sizes
- **Streaming progress**: Real-time progress updates during quantization
- **SafeTensors format**: Secure, fast, memory-mappable tensor storage

## Architecture

```
aljabr-safetensor-quantization/
├── QuantizationEngine.java          # Main quantization orchestration
├── QuantConfig.java                 # Configuration builder
├── QuantStats.java                  # Statistics and metrics
├── QuantResult.java                 # Result container
├── QuantizedTensorInfo.java         # Tensor metadata
├── SafeTensorQuantizedWriter.java   # SafeTensors writer
├── SafeTensorQuantizedReader.java   # SafeTensors reader
├── quantizer/
│   ├── Quantizer.java              # Quantizer interface
│   ├── GPTQQuantizer.java          # INT4 GPTQ implementation
│   ├── INT8Quantizer.java          # INT8 implementation
│   └── FP8Quantizer.java           # FP8 E4M3/E5M2 implementation
└── quantizer/
    ├── Quantizer.java              # Quantizer SPI
    ├── GPTQQuantizer.java          # INT4 GPTQ implementation
    ├── INT8Quantizer.java          # INT8 implementation
    └── FP8Quantizer.java           # FP8 E4M3/E5M2 implementation
```

## Usage

### Programmatic API

#### Basic Quantization

```java
@Inject
QuantizationEngine quantizationEngine;

public void quantizeModel() {
    Path inputPath = Paths.get("/models/llama3-8b-fp16");
    Path outputPath = Paths.get("/models/llama3-8b-int4");
    
    QuantConfig config = QuantConfig.int4Gptq();
    QuantResult result = quantizationEngine.quantize(
        inputPath, 
        outputPath, 
        QuantizationEngine.QuantStrategy.INT4,
        config
    );
    
    if (result.isSuccess()) {
        System.out.println("Compression: " + result.getStats().getCompressionRatio() + "x");
    }
}
```

#### Advanced Configuration

```java
QuantConfig config = QuantConfig.builder()
    .strategy(QuantizationEngine.QuantStrategy.INT4)
    .groupSize(128)           // Group size for GPTQ
    .bits(4)                  // Target bit width
    .symmetric(false)         // Asymmetric quantization
    .perChannel(true)         // Per-channel scaling
    .actOrder(false)          // Static ordering
    .dampPercent(0.01)        // Hessian dampening
    .numSamples(128)          // Calibration samples
    .seqLen(2048)             // Sequence length
    .descAct(false)           // Activation order
    .build();
```

#### Streaming Progress

```java
Multi<Object> progressStream = quantizationEngine.quantizeWithProgress(
    inputPath, 
    outputPath, 
    QuantizationEngine.QuantStrategy.INT4,
    config
);

progressStream.subscribe().with(
    event -> {
        if (event instanceof QuantizationEngine.QuantProgress progress) {
            System.out.printf("Progress: %s (%.1f%%)%n", 
                progress.message(), progress.percentComplete());
        } else if (event instanceof QuantResult result) {
            System.out.println("Complete: " + result.isSuccess());
        }
    },
    error -> System.err.println("Failed: " + error.getMessage())
);
```

#### Loading Quantized Models

```java
@Inject
DirectInferenceEngine engine;

public void loadQuantizedModel() {
    Path quantizedPath = Paths.get("/models/llama3-8b-int4");
    
    String modelKey = engine.loadQuantizedModel(
        quantizedPath, 
        QuantizationEngine.QuantStrategy.INT4
    );
    
    System.out.println("Loaded quantized model: " + modelKey);
}
```

### HTTP integration boundary

This module exposes reusable Java quantization and Safetensors I/O capabilities; it does not provide HTTP endpoints or transport DTOs. The former REST adapter belongs to the deprecated `aljabr-safetensor-api` artifact and is being retired. Applications that need a quantization HTTP API should expose it from their serving/control-plane layer (for example, AQLI) using the transport-neutral `QuantConfig`, `QuantResult`, and `QuantizationEngine` contracts.

## Configuration Reference

### QuantConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | QuantStrategy | INT4 | Quantization strategy (INT4, INT8, FP8) |
| `bits` | int | 4 | Target bit width (1-8) |
| `groupSize` | int | 128 | Group size for group-wise quantization |
| `symmetric` | boolean | false | Use symmetric quantization |
| `perChannel` | boolean | true | Per-channel vs per-tensor scaling |
| `actOrder` | boolean | false | Use activation ordering (GPTQ) |
| `dampPercent` | double | 0.01 | Hessian dampening for numerical stability |
| `numSamples` | int | 128 | Number of calibration samples |
| `seqLen` | int | 2048 | Sequence length for calibration |
| `descAct` | boolean | false | Use descending activation order |

### Strategy Selection Guide

#### INT4 (GPTQ)
- **Use when**: Maximum compression needed, CPU inference
- **Model sizes**: 7B+ parameters
- **Quality**: Good for most use cases
- **Speed**: Slower quantization, fast inference

#### INT8
- **Use when**: Balanced compression/quality
- **Model sizes**: Any size
- **Quality**: Very good, minimal loss
- **Speed**: Fast quantization, fast inference

#### FP8
- **Use when**: GPU with FP8 tensor cores available
- **Model sizes**: Any size
- **Quality**: Excellent, near-lossless
- **Speed**: Fast quantization, fastest inference on supported hardware

## Integration Points

### DirectInferenceEngine

The quantization module integrates seamlessly with `DirectInferenceEngine`:

```java
@Inject
DirectInferenceEngine engine;

// Load quantized model directly
String key = engine.loadQuantizedModel(
    Paths.get("/models/quantized"),
    QuantizationEngine.QuantStrategy.INT4
);

// Access quantization engine
QuantizationEngine qe = engine.getQuantizationEngine();
```

### ModelHotSwapManager

Hot-swap quantized models with zero downtime:

```java
@Inject
ModelHotSwapManager hotSwapManager;

public void upgradeModel() {
    hotSwapManager.beginSwap(
        "llama3-8b",
        oldPath,
        newPath, // Can be quantized model
        null
    ).subscribe().with(...);
}
```

## Performance Considerations

### Quantization Time

| Model Size | INT4 | INT8 | FP8 |
|------------|------|------|-----|
| 3B | ~30s | ~15s | ~20s |
| 7B | ~90s | ~45s | ~60s |
| 13B | ~180s | ~90s | ~120s |
| 70B | ~900s | ~450s | ~600s |

*Times approximate on NVIDIA A100 GPU*

### Memory Requirements

- **INT4**: ~1.5x model size during quantization
- **INT8**: ~1.3x model size during quantization
- **FP8**: ~1.4x model size during quantization

### Inference Performance

| Strategy | CPU Speedup | GPU Speedup | Memory Reduction |
|----------|-------------|-------------|------------------|
| INT4 | 2-3x | 1.5-2x | 75% |
| INT8 | 1.5-2x | 1.3-1.8x | 50% |
| FP8 | N/A | 2-3x (H100) | 50% |

## Testing

### Unit Tests

```bash
mvn test -pl inference-aljabr/extension/runner/safetensor/aljabr-safetensor-quantization
```

### Integration Tests

```bash
mvn verify -pl inference-aljabr/extension/runner/safetensor/aljabr-safetensor-quantization -Pintegration
```

## Troubleshooting

### Common Issues

#### Out of Memory During Quantization
- Reduce `numSamples` in configuration
- Use smaller `seqLen`
- Ensure sufficient GPU memory

#### Poor Quality After Quantization
- Try INT8 instead of INT4
- Increase `groupSize` for INT4
- Enable `actOrder` for better accuracy
- Use FP8 if hardware supports it

#### Slow Quantization
- Use GPU acceleration when available
- Reduce `numSamples` for faster calibration
- Consider INT8 for faster processing

## Future Enhancements

- [ ] AWQ (Activation-aware Weight Quantization)
- [ ] SmoothQuant for activation quantization
- [ ] Mixed-precision quantization
- [ ] Automatic calibration dataset generation
- [ ] Quantization-aware training (QAT) support
- [ ] Neural architecture search for optimal quantization

## References

- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [FP8 NVIDIA H100](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

## License

Apache 2.0 - See LICENSE file for details.
