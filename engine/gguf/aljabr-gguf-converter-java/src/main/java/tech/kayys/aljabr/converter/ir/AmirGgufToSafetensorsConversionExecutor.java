package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter.TensorSpec;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ConversionExecutionResult;
import tech.kayys.aljabr.spi.model.ir.ConversionPlan;
import tech.kayys.aljabr.spi.model.ir.ConversionRequest;
import tech.kayys.aljabr.spi.model.ir.ModelConversionExecutor;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.PayloadReference;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Converts GGUF tensor payloads into Safetensors using bounded-memory streaming.
 *
 * <p>F32/F16/BF16/F64 and integer scalar payloads are copied without changing
 * their representation. MXFP4, NVFP4, TQ1_0, TQ2_0, IQ1_M, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q2_K, Q3_K, Q4_K, Q4_0, Q4_1, Q5_0, Q5_1, Q5_K, Q6_K, and Q8_0 are dequantized to F32. Other quantized
 * GGML encodings are rejected until a verified decoder is available.
 *
 * <p>GGUF scalar and array metadata is retained in a versioned, type-tagged
 * Safetensors metadata envelope. Opaque AMIR extensions are retained as
 * namespaced strings. Versioned envelopes preserve typed assets and
 * components without graph bindings; graph collections and graph-bound
 * components remain explicitly reported.
 */
public final class AmirGgufToSafetensorsConversionExecutor implements ModelConversionExecutor {

    private static final String SOURCE_FORMAT = "gguf";
    private static final String TARGET_FORMAT = "safetensors";
    private static final int COPY_BUFFER_BYTES = 1024 * 1024;
    private static final int BLOCKS_PER_BATCH = 1024;
    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    private static final int[] IQ4_NL_VALUES = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };

    @Override
    public boolean supports(String sourceFormat, String targetFormat) {
        return SOURCE_FORMAT.equalsIgnoreCase(sourceFormat)
                && TARGET_FORMAT.equalsIgnoreCase(targetFormat);
    }

    @Override
    public ConversionExecutionResult execute(
            ModelDescriptor model, ConversionPlan plan, ConversionRequest request) throws IOException {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        Objects.requireNonNull(request, "request must not be null");
        validatePlan(model, plan, request);

        if (!"file".equalsIgnoreCase(request.output().getScheme())) {
            throw new IOException("Safetensors executor requires a file: output URI: " + request.output());
        }
        if (!request.options().isEmpty()) {
            throw new IOException("Unsupported Safetensors conversion options: " + request.options().keySet());
        }

        Path target = Path.of(request.output()).toAbsolutePath().normalize();
        List<TensorSpec> specs = new ArrayList<>(model.tensors().size());
        long outputBytes = 0L;
        for (TensorDescriptor tensor : model.tensors()) {
            PayloadReference payload = tensor.payload();
            if (payload == null) {
                throw new IOException("Tensor '" + tensor.name() + "' has no GGUF payload reference");
            }
            if (payload.offsetBasis() != PayloadReference.OffsetBasis.CONTAINER) {
                throw new IOException("GGUF payload references must use CONTAINER offsets: " + tensor.name());
            }
            Path source = filePath(payload.source());
            if (target.equals(source)) {
                throw new IOException("Output must not replace a source GGUF container: " + source);
            }
            TensorEncoding encoding = encoding(tensor);
            validatePayload(tensor, payload, source, encoding);
            try {
                outputBytes = Math.addExact(outputBytes, encoding.outputBytes(tensor));
            } catch (ArithmeticException overflow) {
                throw new IOException("Total Safetensors payload size overflows long", overflow);
            }
            specs.add(new TensorSpec(
                    tensor.name(),
                    encoding.safetensorsDtype(),
                    safetensorsShape(tensor),
                    encoding.outputBytes(tensor),
                    channel -> writePayload(payload, source, tensor, encoding, channel)));
        }

        Map<String, String> metadata = metadata(model);
        Path parent = target.getParent();
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, "aljabr-gguf-conversion-", ".partial");
        boolean committed = false;
        try {
            StreamingSafetensorWriter.write(temporary, specs, metadata);
            try {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
            }
            committed = true;
        } finally {
            if (!committed) {
                Files.deleteIfExists(temporary);
            }
        }

        return new ConversionExecutionResult(
                SOURCE_FORMAT, TARGET_FORMAT, target.toUri(), specs.size(), outputBytes,
                plan.compatibility(), untransferredKeys(model));
    }

    private static void validatePlan(ModelDescriptor model, ConversionPlan plan, ConversionRequest request)
            throws IOException {
        if (!SOURCE_FORMAT.equalsIgnoreCase(model.sourceFormat())
                || !SOURCE_FORMAT.equalsIgnoreCase(plan.sourceFormat())
                || !TARGET_FORMAT.equalsIgnoreCase(plan.targetFormat())
                || !plan.sourceIdentity().equals(model.identity())) {
            throw new IOException("Conversion plan does not match this GGUF model and Safetensors target");
        }
        CompatibilityReport.ConversionFidelity fidelity = plan.compatibility().fidelity();
        if (fidelity == CompatibilityReport.ConversionFidelity.UNSUPPORTED) {
            throw new IOException("Conversion plan is UNSUPPORTED: " + plan.compatibility().diagnostics());
        }
        boolean hasLossyQuantization = model.tensors().stream().anyMatch(tensor ->
                isDequantizedType(tensor.dataType()));
        boolean hasFlattenedLegacyMetadata = model.extensions().keySet().stream()
                .anyMatch(key -> key.startsWith("gguf.metadata."));
        boolean partial = hasUnrepresentableLayers(model)
                || model.architecture().equalsIgnoreCase("unknown")
                || hasLossyQuantization || hasFlattenedLegacyMetadata;
        if (partial && fidelity != CompatibilityReport.ConversionFidelity.PARTIAL) {
            throw new IOException("Dequantization, legacy flattened metadata, or unrepresented AMIR layers require a PARTIAL plan");
        }
        if (fidelity == CompatibilityReport.ConversionFidelity.PARTIAL && !request.acknowledgePartial()) {
            throw new IOException("GGUF-to-Safetensors conversion is PARTIAL; explicit acknowledgement is required");
        }
    }

    private static TensorEncoding encoding(TensorDescriptor tensor) throws IOException {
        String dtype = tensor.dataType().toUpperCase(Locale.ROOT);
        return switch (dtype) {
            case "F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64" ->
                    TensorEncoding.raw(dtype, bytesPerElement(dtype));
            case "MXFP4" -> TensorEncoding.dequantized("MXFP4", 17, 32);
            case "NVFP4" -> TensorEncoding.dequantized("NVFP4", 68, 64);
            case "TQ1_0" -> TensorEncoding.dequantized("TQ1_0", 54, 256);
            case "TQ2_0" -> TensorEncoding.dequantized("TQ2_0", 66, 256);
            case "IQ1_M" -> TensorEncoding.dequantized("IQ1_M", 56, 256);
            case "IQ1_S" -> TensorEncoding.dequantized("IQ1_S", 50, 256);
            case "IQ2_XXS" -> TensorEncoding.dequantized("IQ2_XXS", 66, 256);
            case "IQ2_XS" -> TensorEncoding.dequantized("IQ2_XS", 74, 256);
            case "IQ2_S" -> TensorEncoding.dequantized("IQ2_S", 82, 256);
            case "IQ3_S" -> TensorEncoding.dequantized("IQ3_S", 110, 256);
            case "IQ3_XXS" -> TensorEncoding.dequantized("IQ3_XXS", 98, 256);
            case "IQ4_NL" -> TensorEncoding.dequantized("IQ4_NL", 18, 32);
            case "IQ4_XS" -> TensorEncoding.dequantized("IQ4_XS", 136, 256);
            case "Q2_0" -> TensorEncoding.dequantized("Q2_0", 18, 64);
            case "Q2_K" -> TensorEncoding.dequantized("Q2_K", 84, 256);
            case "Q3_K" -> TensorEncoding.dequantized("Q3_K", 110, 256);
            case "Q4_K" -> TensorEncoding.dequantized("Q4_K", 144, 256);
            case "Q5_K" -> TensorEncoding.dequantized("Q5_K", 176, 256);
            case "Q4_0" -> TensorEncoding.dequantized("Q4_0", 18, 32);
            case "Q4_1" -> TensorEncoding.dequantized("Q4_1", 20, 32);
            case "Q5_0" -> TensorEncoding.dequantized("Q5_0", 22, 32);
            case "Q5_1" -> TensorEncoding.dequantized("Q5_1", 24, 32);
            case "Q8_0" -> TensorEncoding.dequantized("Q8_0", 34, 32);
            case "Q6_K" -> TensorEncoding.dequantized("Q6_K", 210, 256);
            default -> throw new IOException("No verified GGUF-to-Safetensors decoder for dtype '"
                    + tensor.dataType() + "' (tensor " + tensor.name() + ")");
        };
    }

    private static boolean isDequantizedType(String dtype) {
        return switch (dtype.toUpperCase(Locale.ROOT)) {
            case "MXFP4", "NVFP4", "TQ1_0", "TQ2_0", "IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0", "Q2_K", "Q3_K", "Q4_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0" -> true;
            default -> false;
        };
    }

    private static int bytesPerElement(String dtype) {
        return switch (dtype) {
            case "I8" -> 1;
            case "F16", "BF16", "I16" -> 2;
            case "F32", "I32" -> 4;
            case "F64", "I64" -> 8;
            default -> throw new IllegalArgumentException("Unsupported scalar dtype " + dtype);
        };
    }

    private static void validatePayload(
            TensorDescriptor tensor, PayloadReference payload, Path source, TensorEncoding encoding)
            throws IOException {
        long elements = 1L;
        try {
            if (tensor.shape().isEmpty() || tensor.shape().stream().anyMatch(d -> d == null || d <= 0L)) {
                throw new IOException("Tensor dimensions must be positive: " + tensor.name());
            }
            for (long dimension : tensor.shape()) {
                elements = Math.multiplyExact(elements, dimension);
            }
            if (encoding.dequantized() && elements % encoding.elementsPerBlock() != 0L) {
                throw new IOException("Tensor '" + tensor.name() + "' element count must be divisible by "
                        + encoding.elementsPerBlock() + " for " + encoding.ggmlType());
            }
            long expectedInput = encoding.inputBytes(elements);
            long expectedOutput = encoding.outputBytes(elements);
            if (payload.length() != expectedInput) {
                throw new IOException("Tensor '" + tensor.name() + "' declares " + payload.length()
                        + " input bytes but " + encoding.ggmlType() + " requires " + expectedInput);
            }
            long end = Math.addExact(payload.offset(), payload.length());
            if (!Files.isRegularFile(source) || end > Files.size(source)) {
                throw new IOException("Tensor '" + tensor.name() + "' payload range exceeds source file");
            }
            if (expectedOutput <= 0L) {
                throw new IOException("Tensor output size must be positive: " + tensor.name());
            }
        } catch (ArithmeticException overflow) {
            throw new IOException("Tensor shape or payload size overflows for " + tensor.name(), overflow);
        }
    }

    private static void writePayload(
            PayloadReference payload, Path source, TensorDescriptor tensor,
            TensorEncoding encoding, WritableByteChannel output) throws IOException {
        try (FileChannel input = FileChannel.open(source, StandardOpenOption.READ)) {
            if (encoding.dequantized()) {
                dequantize(input, payload.offset(), tensor.shape(), encoding, output);
            } else {
                copy(input, payload.offset(), payload.length(), output);
            }
        }
    }

    private static void copy(FileChannel input, long offset, long length, WritableByteChannel output)
            throws IOException {
        ByteBuffer buffer = ByteBuffer.allocateDirect(COPY_BUFFER_BYTES);
        long position = offset;
        long remaining = length;
        while (remaining > 0L) {
            checkInterrupted();
            buffer.clear();
            buffer.limit((int) Math.min(buffer.capacity(), remaining));
            int read = input.read(buffer, position);
            if (read < 0) throw new IOException("Unexpected end of GGUF tensor payload");
            if (read == 0) continue;
            buffer.flip();
            writeFully(output, buffer);
            position += read;
            remaining -= read;
        }
    }

    private static void dequantize(
            FileChannel input, long offset, List<Long> shape, TensorEncoding encoding,
            WritableByteChannel output) throws IOException {
        String type = encoding.ggmlType();
        long elements = 1L;
        for (long dimension : shape) elements = Math.multiplyExact(elements, dimension);
        int blockBytes = encoding.scalarBytes();
        int valuesPerBlock = encoding.elementsPerBlock();
        long blocks = elements / valuesPerBlock;
        byte[] source = new byte[blockBytes * BLOCKS_PER_BATCH];
        ByteBuffer values = ByteBuffer.allocateDirect(valuesPerBlock * Float.BYTES * BLOCKS_PER_BATCH)
                .order(ByteOrder.LITTLE_ENDIAN);
        long block = 0L;
        while (block < blocks) {
            checkInterrupted();
            int count = (int) Math.min(BLOCKS_PER_BATCH, blocks - block);
            int bytes = count * blockBytes;
            ByteBuffer packed = ByteBuffer.wrap(source, 0, bytes);
            readFully(input, packed, offset + block * blockBytes);
            values.clear();
            for (int i = 0; i < count; i++) {
                int base = i * blockBytes;
                if (type.equals("MXFP4")) {
                    dequantizeMXFP4Block(source, base, values);
                    continue;
                }
                if (type.equals("NVFP4")) {
                    dequantizeNVFP4Block(source, base, values);
                    continue;
                }
                if (type.equals("TQ1_0")) {
                    dequantizeTQ1Block(source, base, values);
                    continue;
                }
                if (type.equals("TQ2_0")) {
                    dequantizeTQ2Block(source, base, values);
                    continue;
                }
                if (type.equals("IQ1_M")) {
                    dequantizeIQ1MBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ1_S")) {
                    dequantizeIQ1SBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ2_XXS")) {
                    dequantizeIQ2XXSBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ2_XS")) {
                    dequantizeIQ2XSBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ2_S")) {
                    dequantizeIQ2SBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ3_XXS")) {
                    dequantizeIQ3XXSBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ3_S")) {
                    dequantizeIQ3SBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ4_NL")) {
                    dequantizeIQ4NLBlock(source, base, values);
                    continue;
                }
                if (type.equals("IQ4_XS")) {
                    dequantizeIQ4XSBlock(source, base, values);
                    continue;
                }
                if (type.equals("Q2_K")) {
                    dequantizeQ2KBlock(source, base, values, i * valuesPerBlock);
                    continue;
                }
                if (type.equals("Q3_K")) {
                    dequantizeQ3KBlock(source, base, values, i * valuesPerBlock);
                    continue;
                }
                if (type.equals("Q4_K")) {
                    dequantizeQ4KBlock(source, base, values, i * valuesPerBlock);
                    continue;
                }
                if (type.equals("Q5_K")) {
                    dequantizeQ5KBlock(source, base, values, i * valuesPerBlock);
                    continue;
                }
                if (type.equals("Q6_K")) {
                    dequantizeQ6KBlock(source, base, values, i * valuesPerBlock);
                    continue;
                }
                float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
                if (type.equals("Q2_0")) {
                    for (int j = 0; j < 64; j++) {
                        int packedByte = source[base + 2 + (j >>> 2)] & 0xff;
                        int quantized = (packedByte >>> ((j & 3) * 2)) & 0x03;
                        values.putFloat((quantized - 1) * scale);
                    }
                    continue;
                }
                if (type.equals("Q8_0")) {
                    for (int j = 0; j < 32; j++) {
                        values.putFloat(source[base + 2 + j] * scale);
                    }
                    continue;
                }

                float minimum = switch (type) {
                    case "Q4_1", "Q5_1" -> halfToFloat(source[base + 2] & 0xff, source[base + 3] & 0xff);
                    default -> 0.0f;
                };
                int highBitsOffset = type.equals("Q5_0") ? base + 2 : base + 4;
                int quantizedValuesOffset = switch (type) {
                    case "Q4_0" -> base + 2;
                    case "Q4_1" -> base + 4;
                    case "Q5_0" -> base + 6;
                    case "Q5_1" -> base + 8;
                    default -> throw new IOException("Unsupported dequantization type: " + type);
                };
                long highBits = type.startsWith("Q5") ? readUInt32LE(source, highBitsOffset) : 0L;
                for (int j = 0; j < 16; j++) {
                    int packedValue = source[quantizedValuesOffset + j] & 0xff;
                    int quantized = packedValue & 0x0f;
                    if (type.startsWith("Q5")) quantized |= (int) ((highBits >>> j) & 1L) << 4;
                    values.putFloat(dequantizeValue(type, quantized, scale, minimum));
                }
                for (int j = 0; j < 16; j++) {
                    int packedValue = source[quantizedValuesOffset + j] & 0xff;
                    int quantized = packedValue >>> 4;
                    if (type.startsWith("Q5")) quantized |= (int) ((highBits >>> (j + 16)) & 1L) << 4;
                    values.putFloat(dequantizeValue(type, quantized, scale, minimum));
                }
            }
            if (type.equals("Q2_K") || type.equals("Q3_K") || type.equals("Q4_K") || type.equals("Q5_K") || type.equals("Q6_K")) {
                values.position(0);
                values.limit(count * valuesPerBlock * Float.BYTES);
            } else {
                values.flip();
            }
            writeFully(output, values);
            block += count;
        }
    }

    /**
     * GGML block_iq2_s layout: fp16 scale, 32 grid-index bytes, 32 packed sign
     * bytes, eight grid-index high-bit bytes, and eight packed scale bytes.
     * Each 32-value group contains four eight-value grid vectors; the low and
     * high scale nibbles apply to the first and second pair of vectors.
     */
    private static void dequantizeIQ2SBlock(byte[] source, int base, ByteBuffer output) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int qsOffset = base + 2;
        int signsOffset = qsOffset + 32;
        int qhOffset = base + 66;
        int scalesOffset = base + 74;

        for (int group = 0; group < 8; group++) {
            int packedScales = source[scalesOffset + group] & 0xff;
            float firstScale = scale * (0.5f + (packedScales & 0x0f)) * 0.25f;
            float secondScale = scale * (0.5f + (packedScales >>> 4)) * 0.25f;
            int highBits = source[qhOffset + group] & 0xff;
            for (int vector = 0; vector < 4; vector++) {
                int gridIndex = (source[qsOffset + group * 4 + vector] & 0xff)
                        | ((highBits << (8 - 2 * vector)) & 0x300);
                int signs = source[signsOffset + group * 4 + vector] & 0xff;
                float vectorScale = vector < 2 ? firstScale : secondScale;
                for (int lane = 0; lane < 8; lane++) {
                    float sign = (signs & (1 << lane)) != 0 ? -1.0f : 1.0f;
                    output.putFloat(vectorScale * Iq2SGrid.value(gridIndex, lane) * sign);
                }
            }
        }
    }

    /**
     * GGML block_iq2_xs layout: fp16 scale, 32 little-endian 16-bit
     * grid/sign words, then eight packed scale bytes. Each scale byte
     * supplies two four-bit subgroup scales.
     */
    private static void dequantizeIQ2XSBlock(byte[] source, int base, ByteBuffer output) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int qsOffset = base + 2;
        int scalesOffset = base + 66;
        for (int group = 0; group < 8; group++) {
            int packedScales = source[scalesOffset + group] & 0xff;
            float[] groupScales = {
                    scale * (0.5f + (packedScales & 0x0f)) * 0.25f,
                    scale * (0.5f + (packedScales >>> 4)) * 0.25f
            };
            for (int vector = 0; vector < 4; vector++) {
                int wordOffset = qsOffset + (group * 4 + vector) * 2;
                int packed = (source[wordOffset] & 0xff) | ((source[wordOffset + 1] & 0xff) << 8);
                int gridIndex = packed & 0x1ff;
                int signs = decodeIQ3SSignMask((packed >>> 9) & 0x7f);
                float vectorScale = groupScales[vector / 2];
                for (int lane = 0; lane < 8; lane++) {
                    float sign = (signs & (1 << lane)) != 0 ? -1.0f : 1.0f;
                    output.putFloat(vectorScale * Iq2XSGrid.value(gridIndex, lane) * sign);
                }
            }
        }
    }

    /**
     * Decodes one 32-value MXFP4 block. GGML stores an E8M0 scale followed by
     * 16 bytes containing low-nibble values 0..15 and high-nibble values 16..31.
     * The shared E2M1 codebook is stored doubled, so the scale is E8M0 / 2.
     */
    private static void dequantizeMXFP4Block(byte[] source, int base, ByteBuffer output) {
        float scale = e8m0ToFloatHalf(source[base] & 0xff);
        for (int lane = 0; lane < 16; lane++) {
            int packed = source[base + 1 + lane] & 0xff;
            output.putFloat(MXFP4_VALUES[packed & 0x0f] * scale);
        }
        for (int lane = 0; lane < 16; lane++) {
            int packed = source[base + 1 + lane] & 0xff;
            output.putFloat(MXFP4_VALUES[packed >>> 4] * scale);
        }
    }

    /** GGML's E8M0-to-F32 conversion, divided by two for the doubled E2M1 codebook. */
    private static float e8m0ToFloatHalf(int exponent) {
        int bits = exponent < 2
                ? 0x00200000 << exponent
                : (exponent - 1) << 23;
        return Float.intBitsToFloat(bits);
    }

    /**
     * Decodes one 64-value GGML NVFP4 block: four UE4M3 sub-block scales followed
     * by 32 packed E2M1 bytes. Each scale covers 16 values; the shared codebook
     * is doubled, so the unsigned E4M3 scale is converted to its half-value.
     */
    private static void dequantizeNVFP4Block(byte[] source, int base, ByteBuffer output)
            throws IOException {
        for (int subBlock = 0; subBlock < 4; subBlock++) {
            int scaleByte = source[base + subBlock] & 0xff;
            float scale = ue4m3ToFloatHalf(scaleByte);
            int quantizedBase = base + 4 + subBlock * 8;
            for (int lane = 0; lane < 8; lane++) {
                int packed = source[quantizedBase + lane] & 0xff;
                output.putFloat(MXFP4_VALUES[packed & 0x0f] * scale);
            }
            for (int lane = 0; lane < 8; lane++) {
                int packed = source[quantizedBase + lane] & 0xff;
                output.putFloat(MXFP4_VALUES[packed >>> 4] * scale);
            }
        }
    }

    /**
     * GGML UE4M3 scale conversion. The stored unsigned E4M3 scale is halved
     * because the shared E2M1 codebook stores doubled values. GGML maps 0 and
     * the reserved 0x7f NaN encoding to zero.
     */
    private static float ue4m3ToFloatHalf(int value) {
        if (value == 0 || value == 0x7f) {
            return 0.0f;
        }
        int exponent = (value >>> 3) & 0x0f;
        int mantissa = value & 0x07;
        float raw = exponent == 0
                ? Math.scalb((float) mantissa, -9)
                : Math.scalb(1.0f + mantissa / 8.0f, exponent - 7);
        return raw * 0.5f;
    }

    /** Decodes one 256-value TQ1_0 block using GGML's base-3 packed ternary digits. */
    private static void dequantizeTQ1Block(byte[] source, int base, ByteBuffer values) {
        float scale = halfToFloat(source[base + 52] & 0xff, source[base + 53] & 0xff);
        int highBase = base + 48;
        int[] powersOfThree = {1, 3, 9, 27, 81};

        // GGML emits the first 32 low-plane bytes, then the final 16-byte group.
        for (int groupStart = 0; groupStart < 48; groupStart += (groupStart == 0 ? 32 : 16)) {
            int groupLength = groupStart == 0 ? 32 : 16;
            for (int plane = 0; plane < 5; plane++) {
                for (int lane = 0; lane < groupLength; lane++) {
                    int packed = source[base + groupStart + lane] & 0xff;
                    int digit = (packed * powersOfThree[plane]) & 0xff;
                    int ternary = (digit * 3) >>> 8;
                    values.putFloat((ternary - 1) * scale);
                }
            }
        }
        // Four high-plane bytes carry four additional ternary values per plane.
        for (int plane = 0; plane < 4; plane++) {
            for (int lane = 0; lane < 4; lane++) {
                int packed = source[highBase + lane] & 0xff;
                int digit = (packed * powersOfThree[plane]) & 0xff;
                int ternary = (digit * 3) >>> 8;
                values.putFloat((ternary - 1) * scale);
            }
        }
    }

    /** Decodes one 256-value TQ2_0 block in GGML's four bit-plane groups. */
    private static void dequantizeTQ2Block(byte[] source, int base, ByteBuffer values) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int quantizedBase = base + 2;
        for (int byteGroup = 0; byteGroup < 64; byteGroup += 32) {
            for (int plane = 0; plane < 4; plane++) {
                int shift = plane * 2;
                for (int lane = 0; lane < 32; lane++) {
                    int quantized = ((source[quantizedBase + byteGroup + lane] & 0xff) >>> shift) & 0x03;
                    values.putFloat((quantized - 1) * scale);
                }
            }
        }
    }

    /** Decodes IQ1_M using IQ1_S grid vectors and the mixed-scale block header. */
    private static void dequantizeIQ1MBlock(byte[] source, int base, ByteBuffer values) {
        int scalesBase = base + 48;
        int scale0 = readUInt16LE(source, scalesBase);
        int scale1 = readUInt16LE(source, scalesBase + 2);
        int scale2 = readUInt16LE(source, scalesBase + 4);
        int scale3 = readUInt16LE(source, scalesBase + 6);
        int packedScale = (scale0 >>> 12)
                | ((scale1 >>> 8) & 0x00f0)
                | ((scale2 >>> 4) & 0x0f00)
                | (scale3 & 0xf000);
        float scale = halfToFloat(packedScale & 0xff, (packedScale >>> 8) & 0xff);

        int qsBase = base;
        int qhBase = base + 32;
        for (int subBlock = 0; subBlock < 8; subBlock++) {
            int packedScaleWord = readUInt16LE(source, scalesBase + (subBlock / 2) * 2);
            int shift = 6 * (subBlock % 2);
            float firstScale = scale * (2 * ((packedScaleWord >>> shift) & 0x07) + 1);
            float secondScale = scale * (2 * ((packedScaleWord >>> (shift + 3)) & 0x07) + 1);
            int qh0 = source[qhBase + subBlock * 2] & 0xff;
            int qh1 = source[qhBase + subBlock * 2 + 1] & 0xff;
            for (int vector = 0; vector < 4; vector++) {
                int packedHigh = vector < 2 ? qh0 : qh1;
                int highShift = (vector & 1) * 4;
                int signMask = (vector & 1) == 0 ? 0x08 : 0x80;
                int gridIndex = (source[qsBase + subBlock * 4 + vector] & 0xff)
                        | (((packedHigh >>> highShift) & 0x07) << 8);
                float delta = (packedHigh & signMask) != 0 ? -0.125f : 0.125f;
                float vectorScale = vector < 2 ? firstScale : secondScale;
                for (int element = 0; element < 8; element++) {
                    values.putFloat(vectorScale * (Iq1SGrid.value(gridIndex, element) + delta));
                }
            }
        }
    }

    /** Decodes one 256-value IQ1_S block using canonical 8-value grid vectors. */
    private static void dequantizeIQ1SBlock(byte[] source, int base, ByteBuffer values) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int qsBase = base + 2;
        int qhBase = qsBase + 32;
        for (int subBlock = 0; subBlock < 8; subBlock++) {
            int packed = (source[qhBase + subBlock * 2] & 0xff)
                    | ((source[qhBase + subBlock * 2 + 1] & 0xff) << 8);
            float subScale = scale * (2 * ((packed >>> 12) & 0x07) + 1);
            float delta = (packed & 0x8000) != 0 ? -0.125f : 0.125f;
            for (int group = 0; group < 4; group++) {
                int gridIndex = (source[qsBase + subBlock * 4 + group] & 0xff)
                        | (((packed >>> (group * 3)) & 0x07) << 8);
                for (int element = 0; element < 8; element++) {
                    values.putFloat(subScale * (Iq1SGrid.value(gridIndex, element) + delta));
                }
            }
        }
    }

    private static void dequantizeIQ2XXSBlock(byte[] source, int base, ByteBuffer output) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int qsOffset = base + 2;
        for (int group = 0; group < 8; group++) {
            int groupOffset = qsOffset + group * 8;
            long packedSignsAndScale = readUInt32LE(source, groupOffset + 4);
            float groupScale = scale * (0.5f + ((packedSignsAndScale >>> 28) & 0x0fL)) * 0.25f;
            for (int vector = 0; vector < 4; vector++) {
                int gridIndex = source[groupOffset + vector] & 0xff;
                int signs = decodeIQ3SSignMask((int) ((packedSignsAndScale >>> (vector * 7)) & 0x7fL));
                for (int lane = 0; lane < 8; lane++) {
                    float sign = (signs & (1 << lane)) != 0 ? -1.0f : 1.0f;
                    output.putFloat(groupScale * Iq2XXSGrid.value(gridIndex, lane) * sign);
                }
            }
        }
    }

    /**
     * GGML block_iq3_xxs layout: fp16 block scale followed by 96 bytes.
     * The first 64 payload bytes contain eight groups of eight grid indices;
     * the final 32 bytes pack each group's four sign codes and scale nibble.
     */
    private static void dequantizeIQ3XXSBlock(byte[] source, int base, ByteBuffer output) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int qsOffset = base + 2;
        int scalesAndSignsOffset = qsOffset + 64;
        for (int group = 0; group < 8; group++) {
            int aux = (source[scalesAndSignsOffset + group * 4] & 0xff)
                    | ((source[scalesAndSignsOffset + group * 4 + 1] & 0xff) << 8)
                    | ((source[scalesAndSignsOffset + group * 4 + 2] & 0xff) << 16)
                    | ((source[scalesAndSignsOffset + group * 4 + 3] & 0xff) << 24);
            float groupScale = scale * (0.5f + ((aux >>> 28) & 0x0f)) * 0.5f;
            for (int pair = 0; pair < 4; pair++) {
                int signs = decodeIQ3SSignMask((aux >>> (pair * 7)) & 0x7f);
                int firstGrid = source[qsOffset + group * 8 + pair * 2] & 0xff;
                int secondGrid = source[qsOffset + group * 8 + pair * 2 + 1] & 0xff;
                writeIQ3XXSGrid(firstGrid, signs, groupScale, output, 0);
                writeIQ3XXSGrid(secondGrid, signs, groupScale, output, 4);
            }
        }
    }

    private static void writeIQ3XXSGrid(
            int gridIndex, int signs, float scale, ByteBuffer output, int signOffset) {
        for (int lane = 0; lane < 4; lane++) {
            float sign = (signs & (1 << (signOffset + lane))) != 0 ? -1.0f : 1.0f;
            output.putFloat(scale * Iq3XXSGrid.value(gridIndex, lane) * sign);
        }
    }

    /**
     * GGML block_iq3_s layout: fp16 scale, 64 low grid-index bytes, eight high-index
     * bytes, 32 packed sign codes, and four bytes of paired four-bit group scales.
     * The canonical 512-entry grid is stored as little-endian packed bytes.
     */
    private static void dequantizeIQ3SBlock(byte[] source, int base, ByteBuffer output) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int qsOffset = base + 2;
        int qhOffset = base + 66;
        int signsOffset = base + 74;
        int scalesOffset = base + 106;
        for (int pair = 0; pair < 4; pair++) {
            int packedScales = source[scalesOffset + pair] & 0xff;
            float firstScale = scale * (1 + 2 * (packedScales & 0x0f));
            float secondScale = scale * (1 + 2 * (packedScales >>> 4));
            int qhFirst = source[qhOffset + pair * 2] & 0xff;
            int qhSecond = source[qhOffset + pair * 2 + 1] & 0xff;
            writeIQ3SGridPair(source, qsOffset + pair * 16, qhFirst,
                    signsOffset + pair * 8, firstScale, output);
            writeIQ3SGridPair(source, qsOffset + pair * 16 + 8, qhSecond,
                    signsOffset + pair * 8 + 4, secondScale, output);
        }
    }

    private static void writeIQ3SGridPair(
            byte[] source, int qsOffset, int qh, int signsOffset, float scale, ByteBuffer output) {
        for (int lane = 0; lane < 4; lane++) {
            int firstIndex = (source[qsOffset + lane * 2] & 0xff)
                    | (((qh << (8 - 2 * lane)) & 0x100));
            int secondIndex = (source[qsOffset + lane * 2 + 1] & 0xff)
                    | (((qh << (7 - 2 * lane)) & 0x100));
            int signMask = decodeIQ3SSignMask(source[signsOffset + lane] & 0x7f);
            for (int value = 0; value < 4; value++) {
                float sign = (signMask & (1 << value)) != 0 ? -1.0f : 1.0f;
                output.putFloat(scale * Iq3SGrid.value(firstIndex, value) * sign);
            }
            for (int value = 0; value < 4; value++) {
                float sign = (signMask & (1 << (value + 4))) != 0 ? -1.0f : 1.0f;
                output.putFloat(scale * Iq3SGrid.value(secondIndex, value) * sign);
            }
        }
    }

    private static int decodeIQ3SSignMask(int encoded) {
        return encoded | ((Integer.bitCount(encoded) & 1) << 7);
    }

    /**
     * GGML block_iq4_nl layout: fp16 scale followed by sixteen packed nibble pairs.
     * The nonlinear codebook maps each nibble to a signed integer before scaling.
     */
    private static void dequantizeIQ4NLBlock(byte[] source, int base, ByteBuffer output) {
        float scale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int quantizedOffset = base + 2;
        for (int i = 0; i < 16; i++) {
            int packed = source[quantizedOffset + i] & 0xff;
            output.putFloat(scale * IQ4_NL_VALUES[packed & 0x0f]);
        }
        for (int i = 0; i < 16; i++) {
            int packed = source[quantizedOffset + i] & 0xff;
            output.putFloat(scale * IQ4_NL_VALUES[packed >>> 4]);
        }
    }

    /**
     * GGML block_iq4_xs layout: fp16 super-block scale, uint16 packed scale high bits,
     * four bytes of packed scale low nibbles, then 128 bytes of IQ4_NL codebook indices.
     * Each of the eight 32-value groups has a signed six-bit scale (stored with bias 32).
     */
    private static void dequantizeIQ4XSBlock(byte[] source, int base, ByteBuffer output) {
        float superBlockScale = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        int packedHighScales = (source[base + 2] & 0xff) | ((source[base + 3] & 0xff) << 8);
        int lowScalesOffset = base + 4;
        int quantizedOffset = base + 8;
        for (int group = 0; group < 8; group++) {
            int packedLow = source[lowScalesOffset + group / 2] & 0xff;
            int lowScale = (packedLow >>> (4 * (group & 1))) & 0x0f;
            int highScale = (packedHighScales >>> (2 * group)) & 0x03;
            int scale = lowScale | (highScale << 4);
            float groupScale = superBlockScale * (scale - 32);
            int groupQuantizedOffset = quantizedOffset + group * 16;
            for (int i = 0; i < 16; i++) {
                int packed = source[groupQuantizedOffset + i] & 0xff;
                output.putFloat(groupScale * IQ4_NL_VALUES[packed & 0x0f]);
            }
            for (int i = 0; i < 16; i++) {
                int packed = source[groupQuantizedOffset + i] & 0xff;
                output.putFloat(groupScale * IQ4_NL_VALUES[packed >>> 4]);
            }
        }
    }

    /**
     * GGML block_q2_K layout: scales[16], qs[64], fp16 d, fp16 dmin.
     * Four 2-bit planes are unpacked into sequential groups of 32 values.
     */
    private static void dequantizeQ2KBlock(byte[] source, int base, ByteBuffer output, int blockIndex) {
        float d = halfToFloat(source[base + 80] & 0xff, source[base + 81] & 0xff);
        float min = halfToFloat(source[base + 82] & 0xff, source[base + 83] & 0xff);
        int outputBase = blockIndex * 256;
        for (int half = 0; half < 2; half++) {
            int scaleBase = base + half * 8;
            int quantizedBase = base + 16 + half * 32;
            for (int plane = 0; plane < 4; plane++) {
                int shift = plane * 2;
                int outputGroup = outputBase + half * 128 + plane * 32;
                writeQ2KGroup(source, quantizedBase, scaleBase + plane * 2,
                        shift, d, min, output, outputGroup);
                writeQ2KGroup(source, quantizedBase + 16, scaleBase + plane * 2 + 1,
                        shift, d, min, output, outputGroup + 16);
            }
        }
    }

    private static void writeQ2KGroup(
            byte[] source, int quantizedOffset, int scaleOffset, int shift,
            float d, float min, ByteBuffer output, int outputOffset) {
        int scale = source[scaleOffset] & 0xff;
        float dl = d * (scale & 0x0f);
        float ml = min * (scale >>> 4);
        for (int i = 0; i < 16; i++) {
            int quantized = ((source[quantizedOffset + i] & 0xff) >>> shift) & 0x03;
            output.putFloat((outputOffset + i) * Float.BYTES, dl * quantized - ml);
        }
    }

    /**
     * GGML block_q3_K layout: hmask[32], qs[64], scales[12], fp16 d.
     * The 12 packed scale bytes encode sixteen signed 6-bit subgroup scales.
     */
    private static void dequantizeQ3KBlock(byte[] source, int base, ByteBuffer output, int blockIndex) {
        int hmaskOffset = base;
        int quantizedOffset = base + 32;
        int scalesOffset = base + 96;
        float d = halfToFloat(source[base + 108] & 0xff, source[base + 109] & 0xff);
        int[] scales = unpackQ3KScales(source, scalesOffset);
        int outputBase = blockIndex * 256;

        for (int half = 0; half < 2; half++) {
            int qBase = quantizedOffset + half * 32;
            int hBase = hmaskOffset + half * 16;
            for (int plane = 0; plane < 4; plane++) {
                int shift = plane * 2;
                int mask = 1 << plane;
                for (int lane = 0; lane < 2; lane++) {
                    int scaleIndex = half * 8 + plane * 2 + lane;
                    float scale = d * (scales[scaleIndex] - 32);
                    int qLane = qBase + lane * 16;
                    int out = outputBase + half * 128 + plane * 32 + lane * 16;
                    for (int i = 0; i < 16; i++) {
                        int low = ((source[qLane + i] & 0xff) >>> shift) & 0x03;
                        int highMask = source[hBase + lane * 16 + i] & 0xff;
                        int signedQuant = low - ((highMask & mask) != 0 ? 0 : 4);
                        output.putFloat((out + i) * Float.BYTES, scale * signedQuant);
                    }
                }
            }
        }
    }

    private static int[] unpackQ3KScales(byte[] source, int offset) {
        int[] scales = new int[16];
        for (int i = 0; i < 4; i++) {
            int packedLow = source[offset + i] & 0xff;
            int packedHigh = source[offset + i + 4] & 0xff;
            int packedTop = source[offset + i + 8] & 0xff;
            scales[i] = (packedLow & 0x0f) | ((packedTop & 0x03) << 4);
            scales[i + 4] = (packedHigh & 0x0f) | (((packedTop >>> 2) & 0x03) << 4);
            scales[i + 8] = ((packedLow >>> 4) & 0x0f) | (((packedTop >>> 4) & 0x03) << 4);
            scales[i + 12] = ((packedHigh >>> 4) & 0x0f) | (((packedTop >>> 6) & 0x03) << 4);
        }
        return scales;
    }

    /**
     * GGML block_q4_K layout: fp16 d, fp16 dmin, scales[12], qs[128].
     * Eight 32-value groups use unsigned 4-bit values and independent 6-bit
     * scale/minimum factors. GGML emits each pair of groups from low/high nibbles.
     */
    private static void dequantizeQ4KBlock(byte[] source, int base, ByteBuffer output, int blockIndex) {
        float d = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        float dmin = halfToFloat(source[base + 2] & 0xff, source[base + 3] & 0xff);
        int[] scales = new int[8];
        int[] mins = new int[8];
        unpackQ4KScalesAndMins(source, base + 4, scales, mins);

        int outputBase = blockIndex * 256;
        int quantizedBase = base + 16;
        for (int iteration = 0; iteration < 4; iteration++) {
            int firstGroup = iteration * 2;
            float firstScale = d * scales[firstGroup];
            float firstMin = dmin * mins[firstGroup];
            float secondScale = d * scales[firstGroup + 1];
            float secondMin = dmin * mins[firstGroup + 1];
            int quantizedOffset = quantizedBase + iteration * 32;
            int outputOffset = outputBase + iteration * 64;

            for (int i = 0; i < 32; i++) {
                int packed = source[quantizedOffset + i] & 0xff;
                output.putFloat((outputOffset + i) * Float.BYTES,
                        firstScale * (packed & 0x0f) - firstMin);
                output.putFloat((outputOffset + 32 + i) * Float.BYTES,
                        secondScale * (packed >>> 4) - secondMin);
            }
        }
    }

    /**
     * Unpacks GGML get_scale_min_k4's eight 6-bit scales and eight 6-bit minima.
     */
    private static void unpackQ4KScalesAndMins(
            byte[] source, int offset, int[] scales, int[] mins) {
        for (int i = 0; i < 4; i++) {
            int scaleLow = source[offset + i] & 0xff;
            int minLow = source[offset + i + 4] & 0xff;
            int packedHigh = source[offset + i + 8] & 0xff;

            scales[i] = scaleLow & 0x3f;
            mins[i] = minLow & 0x3f;
            scales[i + 4] = (packedHigh & 0x0f) | ((scaleLow >>> 6) << 4);
            mins[i + 4] = ((packedHigh >>> 4) & 0x0f) | ((minLow >>> 6) << 4);
        }
    }

    /**
     * GGML block_q5_K layout: fp16 d, fp16 dmin, scales[12], qh[32], qs[128].
     * The high-bit plane is shared by four 64-value segments; each segment
     * selects a different two-bit pair from each qh byte.
     */
    private static void dequantizeQ5KBlock(byte[] source, int base, ByteBuffer output, int blockIndex) {
        float d = halfToFloat(source[base] & 0xff, source[base + 1] & 0xff);
        float dmin = halfToFloat(source[base + 2] & 0xff, source[base + 3] & 0xff);
        int[] scales = new int[8];
        int[] mins = new int[8];
        unpackQ4KScalesAndMins(source, base + 4, scales, mins);

        int highBitsOffset = base + 16;
        int quantizedOffset = base + 48;
        int outputBase = blockIndex * 256;
        for (int segment = 0; segment < 4; segment++) {
            int lowMask = 1 << (segment * 2);
            int highMask = 2 << (segment * 2);
            int firstGroup = segment * 2;
            float firstScale = d * scales[firstGroup];
            float firstMin = dmin * mins[firstGroup];
            float secondScale = d * scales[firstGroup + 1];
            float secondMin = dmin * mins[firstGroup + 1];
            int qOffset = quantizedOffset + segment * 32;
            int outputOffset = outputBase + segment * 64;

            for (int i = 0; i < 32; i++) {
                int packed = source[qOffset + i] & 0xff;
                int high = source[highBitsOffset + i] & 0xff;
                int lowValue = (packed & 0x0f) | ((high & lowMask) != 0 ? 16 : 0);
                int highValue = (packed >>> 4) | ((high & highMask) != 0 ? 16 : 0);
                output.putFloat((outputOffset + i) * Float.BYTES,
                        firstScale * lowValue - firstMin);
                output.putFloat((outputOffset + 32 + i) * Float.BYTES,
                        secondScale * highValue - secondMin);
            }
        }
    }

    /**
     * GGML block_q6_K layout: ql[128], qh[64], scales[16], fp16 d.
     * Each 128-value half stores four interleaved 32-value lanes.
     */
    private static void dequantizeQ6KBlock(byte[] source, int base, ByteBuffer output, int blockIndex) {
        float d = halfToFloat(source[base + 208] & 0xff, source[base + 209] & 0xff);
        int outputBase = blockIndex * 256;
        for (int half = 0; half < 2; half++) {
            int qlBase = base + half * 64;
            int qhBase = base + 128 + half * 32;
            int scaleBase = base + 192 + half * 8;
            for (int lane = 0; lane < 32; lane++) {
                int packedHigh = source[qhBase + lane] & 0xff;
                int scaleGroup = lane / 16;
                int low0 = source[qlBase + lane] & 0xff;
                int low1 = source[qlBase + lane + 32] & 0xff;
                int q1 = (low0 & 0x0f) | ((packedHigh & 0x03) << 4);
                int q2 = (low1 & 0x0f) | (((packedHigh >>> 2) & 0x03) << 4);
                int q3 = (low0 >>> 4) | (((packedHigh >>> 4) & 0x03) << 4);
                int q4 = (low1 >>> 4) | (((packedHigh >>> 6) & 0x03) << 4);
                putQ6Value(output, outputBase + lane, d, source[scaleBase + scaleGroup], q1);
                putQ6Value(output, outputBase + lane + 32, d, source[scaleBase + scaleGroup + 2], q2);
                putQ6Value(output, outputBase + lane + 64, d, source[scaleBase + scaleGroup + 4], q3);
                putQ6Value(output, outputBase + lane + 96, d, source[scaleBase + scaleGroup + 6], q4);
            }
        }
    }

    private static void putQ6Value(ByteBuffer output, int index, float d, byte scale, int quantized) {
        output.putFloat(index * Float.BYTES, d * scale * (quantized - 32));
    }

    private static float dequantizeValue(String type, int quantized, float scale, float minimum) {
        return switch (type) {
            case "Q4_0" -> (quantized - 8) * scale;
            case "Q4_1", "Q5_1" -> quantized * scale + minimum;
            case "Q5_0" -> (quantized - 16) * scale;
            default -> throw new IllegalArgumentException("Unsupported quantization type: " + type);
        };
    }

    private static int readUInt16LE(byte[] bytes, int offset) {
        return (bytes[offset] & 0xff) | ((bytes[offset + 1] & 0xff) << 8);
    }

    private static long readUInt32LE(byte[] bytes, int offset) {
        return (bytes[offset] & 0xffL)
                | ((bytes[offset + 1] & 0xffL) << 8)
                | ((bytes[offset + 2] & 0xffL) << 16)
                | ((bytes[offset + 3] & 0xffL) << 24);
    }

    private static float halfToFloat(int low, int high) {
        return Float.float16ToFloat((short) (low | (high << 8)));
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) throw new IOException("Unexpected end of GGUF tensor payload");
            if (read == 0) continue;
            position += read;
        }
    }

    private static void writeFully(WritableByteChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) channel.write(buffer);
    }

    private static void checkInterrupted() throws IOException {
        if (Thread.currentThread().isInterrupted()) {
            throw new IOException("GGUF-to-Safetensors conversion interrupted");
        }
    }

    /**
     * GGUF stores dimensions in ggml order (dimension zero is contiguous /
     * fastest-changing). Safetensors consumers use the conventional outermost-
     * to-innermost shape order, so the dimensions must always be reversed.
     *
     * <p>This is a metadata-only transformation: the payload byte sequence is
     * intentionally copied unchanged.
     */
    private static List<Long> safetensorsShape(TensorDescriptor tensor) {
        if (tensor.shape().size() < 2) {
            return tensor.shape();
        }
        List<Long> reversed = new ArrayList<>(tensor.shape());
        java.util.Collections.reverse(reversed);
        return List.copyOf(reversed);
    }

    private static Map<String, String> metadata(ModelDescriptor model) {
        Map<String, String> values = new LinkedHashMap<>();
        values.put("aljabr.source.format", SOURCE_FORMAT);
        values.put("aljabr.source.architecture", model.architecture());
        model.metadata().forEach((key, value) ->
                AmirSourceMetadataCodec.putUnique(values, "aljabr.source.metadata." + key, value,
                        "Safetensors source metadata"));
        model.extensions().forEach((key, value) ->
                AmirSourceMetadataCodec.putUnique(values, "aljabr.source.extension." + key, value,
                        "Safetensors source extension"));
        AmirTypedMetadataCodec.encode(model.typedMetadata()).forEach((key, value) ->
                AmirSourceMetadataCodec.putUnique(values, key, value, "Safetensors typed metadata"));
        AmirTensorTypeCodec.encode(model.tensors()).forEach((key, value) ->
                AmirSourceMetadataCodec.putUnique(values, key, value, "Safetensors typed tensor metadata"));
        AmirModelLayerCodec.encode(model).forEach((key, value) ->
                AmirSourceMetadataCodec.putUnique(values, key, value, "Safetensors model layer metadata"));
        for (TensorDescriptor tensor : model.tensors()) {
            tensor.extensions().forEach((key, value) -> AmirSourceMetadataCodec.putUnique(values,
                    "aljabr.source.tensor." + keyPart(tensor.name()) + "." + keyPart(key), value,
                    "Safetensors tensor extension"));
        }
        return Map.copyOf(values);
    }

    private static String keyPart(String value) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }

    private static boolean hasUnrepresentableLayers(ModelDescriptor model) {
        return model.graphCollection().isPresent()
                || model.componentCollection().filter(collection ->
                        !AmirModelLayerCodec.hasNoGraphBindings(collection)).isPresent();
    }

    private static List<String> untransferredKeys(ModelDescriptor model) {
        List<String> keys = new ArrayList<>();
        model.graphCollection().ifPresent(ignored -> keys.add("model:graphCollection"));
        model.componentCollection().filter(collection ->
                !AmirModelLayerCodec.hasNoGraphBindings(collection))
                .ifPresent(ignored -> keys.add("model:componentCollection"));
        model.extensions().keySet().stream()
                .filter(key -> key.startsWith("gguf.metadata."))
                .sorted()
                .forEach(key -> keys.add("model:legacy-typed-metadata:"
                        + key.substring("gguf.metadata.".length())));
        return List.copyOf(keys);
    }

    private static Path filePath(URI source) throws IOException {
        if (!"file".equalsIgnoreCase(source.getScheme())) {
            throw new IOException("Only file: GGUF payload references are supported: " + source);
        }
        try {
            return Path.of(source).toAbsolutePath().normalize();
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid GGUF payload URI: " + source, invalid);
        }
    }

    private record TensorEncoding(
            String ggmlType, String safetensorsDtype, int scalarBytes, boolean dequantized, int elementsPerBlock) {
        static TensorEncoding raw(String dtype, int scalarBytes) {
            return new TensorEncoding(dtype, dtype, scalarBytes, false, 1);
        }
        static TensorEncoding dequantized(String type, int blockBytes, int elementsPerBlock) {
            return new TensorEncoding(type, "F32", blockBytes, true, elementsPerBlock);
        }
        long inputBytes(long elements) {
            return dequantized ? Math.multiplyExact(elements / elementsPerBlock, scalarBytes)
                    : Math.multiplyExact(elements, scalarBytes);
        }
        long outputBytes(TensorDescriptor tensor) {
            return outputBytes(tensor.shape().stream().mapToLong(Long::longValue)
                    .reduce(1L, Math::multiplyExact));
        }
        long outputBytes(long elements) {
            return Math.multiplyExact(elements, dequantized ? Float.BYTES : scalarBytes);
        }
    }
}
