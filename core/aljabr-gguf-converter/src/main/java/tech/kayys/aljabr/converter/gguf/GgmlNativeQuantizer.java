package tech.kayys.aljabr.converter.gguf;

import tech.kayys.aljabr.gguf.core.GgmlType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Thin FFM binding to ggml's reference quantizer.
 *
 * <p>The Java K-quant encoders are useful as a portability fallback, but ggml's
 * native implementation is the compatibility source of truth for files that are
 * loaded by llama.cpp.</p>
 * @author bhangun
 */
final class GgmlNativeQuantizer {
    private static final String PROP_DISABLED = "aljabr.gguf.native_quant.disable";
    private static final String PROP_LIBRARY = "aljabr.gguf.ggml.library.path";
    private static final String PROP_LIB_DIR = "aljabr.gguf.ggml.libdir";
    private static final String ENV_LIBRARY = "ALJABR_GGML_LIBRARY_PATH";
    private static final String ENV_LIB_DIR = "ALJABR_GGML_LIB_DIR";

    private static volatile Optional<GgmlNativeQuantizer> cached;

    private final MethodHandle quantizeChunk;
    private final Path libraryPath;

    private GgmlNativeQuantizer(MethodHandle quantizeChunk, Path libraryPath) {
        this.quantizeChunk = quantizeChunk;
        this.libraryPath = libraryPath;
    }

    static Optional<GgmlNativeQuantizer> load() {
        Optional<GgmlNativeQuantizer> current = cached;
        if (current != null) {
            return current;
        }
        synchronized (GgmlNativeQuantizer.class) {
            current = cached;
            if (current == null) {
                current = loadOnce();
                cached = current;
            }
            return current;
        }
    }

    Path libraryPath() {
        return libraryPath;
    }

    byte[] quantize(byte[] srcF32, long numElements, long rowElements, GgmlType type) {
        if (numElements <= 0 || rowElements <= 0 || numElements % rowElements != 0) {
            throw new IllegalArgumentException("Invalid quantization shape: elements=" + numElements
                    + ", rowElements=" + rowElements);
        }
        if (rowElements % type.blockSize != 0) {
            throw new IllegalArgumentException("Row width " + rowElements
                    + " is not a multiple of block size " + type.blockSize + " for " + type.label);
        }

        byte[] dst = new byte[Math.toIntExact(type.bytesFor(numElements))];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocate(srcF32.length, 4);
            MemorySegment out = arena.allocate(dst.length, 32);
            MemorySegment.copy(srcF32, 0, src, ValueLayout.JAVA_BYTE, 0, srcF32.length);

            long rows = numElements / rowElements;
            long written = (long) quantizeChunk.invokeExact(
                    type.id,
                    src,
                    out,
                    0L,
                    rows,
                    rowElements,
                    MemorySegment.NULL);

            if (written != dst.length) {
                throw new IllegalStateException("ggml_quantize_chunk wrote " + written
                        + " bytes, expected " + dst.length + " for " + type.label);
            }

            MemorySegment.copy(out, ValueLayout.JAVA_BYTE, 0, dst, 0, dst.length);
            return dst;
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException("Native ggml quantization failed for " + type.label, t);
        }
    }

    private static Optional<GgmlNativeQuantizer> loadOnce() {
        if (Boolean.getBoolean(PROP_DISABLED)) {
            return Optional.empty();
        }

        List<Path> candidates = libraryCandidates();
        List<String> failures = new ArrayList<>();
        for (Path candidate : candidates) {
            if (!Files.isRegularFile(candidate)) {
                continue;
            }
            try {
                preloadKnownDependencies(candidate.getParent());
                SymbolLookup lookup = SymbolLookup.libraryLookup(candidate, Arena.global());
                MethodHandle handle = Linker.nativeLinker().downcallHandle(
                        lookup.find("ggml_quantize_chunk")
                                .orElseThrow(() -> new UnsatisfiedLinkError("Missing ggml_quantize_chunk")),
                        FunctionDescriptor.of(
                                ValueLayout.JAVA_LONG,
                                ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.ADDRESS));
                return Optional.of(new GgmlNativeQuantizer(handle, candidate));
            } catch (Throwable t) {
                failures.add(candidate + ": " + t.getMessage());
            }
        }

        if (Boolean.getBoolean("aljabr.gguf.native_quant.debug") && !failures.isEmpty()) {
            System.err.println("[gguf] Native ggml quantizer unavailable:");
            for (String failure : failures) {
                System.err.println("[gguf]   " + failure);
            }
        }
        return Optional.empty();
    }

    private static List<Path> libraryCandidates() {
        List<Path> out = new ArrayList<>();
        String explicit = firstNonBlank(System.getProperty(PROP_LIBRARY), System.getenv(ENV_LIBRARY));
        if (explicit != null) {
            out.add(Path.of(explicit));
            return out;
        }

        String explicitDir = firstNonBlank(System.getProperty(PROP_LIB_DIR), System.getenv(ENV_LIB_DIR));
        if (explicitDir != null) {
            addNames(out, Path.of(explicitDir));
        }

        Path home = Path.of(System.getProperty("user.home"));
        addNames(out, home.resolve(".aljabr/libs/llama"));
        addNames(out, home.resolve(".aljabr/source/vendor/llama.cpp-gemma4/llama.cpp/build-aljabr/bin"));
        return out;
    }

    private static void addNames(List<Path> out, Path dir) {
        if (System.getProperty("os.name", "").toLowerCase().contains("mac")) {
            out.add(dir.resolve("libggml.dylib"));
            out.add(dir.resolve("libggml.0.dylib"));
        } else if (System.getProperty("os.name", "").toLowerCase().contains("win")) {
            out.add(dir.resolve("ggml.dll"));
        } else {
            out.add(dir.resolve("libggml.so"));
            out.add(dir.resolve("libggml.so.0"));
        }
    }

    private static void preloadKnownDependencies(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) {
            return;
        }

        String suffix;
        if (System.getProperty("os.name", "").toLowerCase().contains("mac")) {
            suffix = ".dylib";
        } else if (System.getProperty("os.name", "").toLowerCase().contains("win")) {
            suffix = ".dll";
        } else {
            suffix = ".so";
        }

        String[] names = {
                "libggml-base",
                "libggml-cpu",
                "libggml-blas",
                "libggml-metal"
        };
        for (String name : names) {
            Path p = dir.resolve(name + suffix);
            if (Files.isRegularFile(p)) {
                try {
                    System.load(p.toAbsolutePath().toString());
                } catch (UnsatisfiedLinkError ignored) {
                    // libraryLookup below reports the actionable failure if this matters
                }
            }
        }
    }

    private static String firstNonBlank(String first, String second) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (second != null && !second.isBlank()) {
            return second;
        }
        return null;
    }
}
