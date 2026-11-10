package tech.kayys.aljabr.gguf.javabackend;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

/**
 * High-performance disk-backed persistent storage for prompt KV-caches.
 * Enables sub-15ms prefill restoration across daemon restarts and CLI invocations.
 */
final class GgufPromptCacheDiskStorage {
    private static final byte[] MAGIC = new byte[] {'A', 'Q', 'L', 'I', 'P', 'K', 'V', '1'};
    private static final int VERSION = 1;

    private GgufPromptCacheDiskStorage() {}

    static Path getCacheDir() {
        String custom = System.getProperty("aljabr.prompt.cache.dir");
        if (custom == null || custom.isBlank()) {
            custom = System.getenv("ALJABR_PROMPT_CACHE_DIR");
        }
        if (custom != null && !custom.isBlank()) {
            return Path.of(custom);
        }
        Path home = Path.of(System.getProperty("user.home"));
        Path aqliDir = home.resolve(".aqli");
        if (Files.exists(aqliDir)) {
            return aqliDir.resolve("cache").resolve("prompt-kv");
        }
        return home.resolve(".aljabr").resolve("cache").resolve("prompt-kv");
    }

    static String computeKey(Path modelPath, int[] tokens) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(modelPath.toAbsolutePath().normalize().toString().getBytes(StandardCharsets.UTF_8));
            try {
                if (Files.isRegularFile(modelPath)) {
                    long size = Files.size(modelPath);
                    long mtime = Files.getLastModifiedTime(modelPath).toMillis();
                    ByteBuffer meta = ByteBuffer.allocate(16);
                    meta.putLong(size).putLong(mtime);
                    md.update(meta.array());
                }
            } catch (Exception ignored) {
            }
            ByteBuffer tBuf = ByteBuffer.allocate(tokens.length * 4);
            for (int t : tokens) {
                tBuf.putInt(t);
            }
            md.update(tBuf.array());
            byte[] digest = md.digest();
            StringBuilder sb = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 missing", e);
        }
    }

    static PromptCacheEntry loadFromDisk(Path cacheDir, Path modelPath, int[] tokens) {
        if (cacheDir == null || modelPath == null || tokens == null || tokens.length == 0) {
            return null;
        }
        String key = computeKey(modelPath, tokens);
        Path file = cacheDir.resolve(key + ".pkv");
        if (!Files.isRegularFile(file)) {
            return null;
        }
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long size = ch.size();
            if (size < 24) {
                return null;
            }
            ByteBuffer buf = ByteBuffer.allocate((int) size).order(ByteOrder.LITTLE_ENDIAN);
            while (buf.hasRemaining()) {
                if (ch.read(buf) < 0) {
                    break;
                }
            }
            buf.flip();

            byte[] magic = new byte[8];
            buf.get(magic);
            if (!Arrays.equals(magic, MAGIC)) {
                return null;
            }
            int version = buf.getInt();
            if (version != VERSION) {
                return null;
            }
            int tokenCount = buf.getInt();
            if (tokenCount != tokens.length) {
                return null;
            }
            int[] diskTokens = new int[tokenCount];
            buf.asIntBuffer().get(diskTokens);
            buf.position(buf.position() + tokenCount * 4);
            if (!Arrays.equals(diskTokens, tokens)) {
                return null;
            }
            int nLayers = buf.getInt();
            int logitsLen = buf.getInt();
            float[] logits = new float[logitsLen];
            buf.asFloatBuffer().get(logits);
            buf.position(buf.position() + logitsLen * 4);

            float[][] k = new float[nLayers][];
            float[][] v = new float[nLayers][];
            for (int l = 0; l < nLayers; l++) {
                int kCount = buf.getInt();
                k[l] = new float[kCount];
                buf.asFloatBuffer().get(k[l]);
                buf.position(buf.position() + kCount * 4);

                int vCount = buf.getInt();
                v[l] = new float[vCount];
                buf.asFloatBuffer().get(v[l]);
                buf.position(buf.position() + vCount * 4);
            }
            return new PromptCacheEntry(tokens, k, v, logits);
        } catch (Throwable ignored) {
            return null;
        }
    }

    static void saveToDiskAsync(Path cacheDir, Path modelPath, PromptCacheEntry entry) {
        if (cacheDir == null || modelPath == null || entry == null) {
            return;
        }
        CompletableFuture.runAsync(() -> {
            try {
                saveToDisk(cacheDir, modelPath, entry);
            } catch (Throwable ignored) {
            }
        });
    }

    static void saveToDisk(Path cacheDir, Path modelPath, PromptCacheEntry entry) throws IOException {
        Files.createDirectories(cacheDir);
        String key = computeKey(modelPath, entry.tokens);
        Path targetFile = cacheDir.resolve(key + ".pkv");
        Path tempFile = cacheDir.resolve(key + ".tmp." + ProcessHandle.current().pid() + "." + System.nanoTime());

        int totalBytes = 8 // magic
                + 4 // version
                + 4 // tokenCount
                + entry.tokens.length * 4
                + 4 // nLayers
                + 4 // logitsLen
                + entry.logits.length * 4;

        int nLayers = entry.k.length;
        for (int l = 0; l < nLayers; l++) {
            totalBytes += 4 + entry.k[l].length * 4;
            totalBytes += 4 + entry.v[l].length * 4;
        }

        ByteBuffer buf = ByteBuffer.allocate(totalBytes).order(ByteOrder.LITTLE_ENDIAN);
        buf.put(MAGIC);
        buf.putInt(VERSION);
        buf.putInt(entry.tokens.length);
        buf.asIntBuffer().put(entry.tokens);
        buf.position(buf.position() + entry.tokens.length * 4);
        buf.putInt(nLayers);
        buf.putInt(entry.logits.length);
        buf.asFloatBuffer().put(entry.logits);
        buf.position(buf.position() + entry.logits.length * 4);

        for (int l = 0; l < nLayers; l++) {
            buf.putInt(entry.k[l].length);
            buf.asFloatBuffer().put(entry.k[l]);
            buf.position(buf.position() + entry.k[l].length * 4);

            buf.putInt(entry.v[l].length);
            buf.asFloatBuffer().put(entry.v[l]);
            buf.position(buf.position() + entry.v[l].length * 4);
        }
        buf.flip();

        try (FileChannel ch = FileChannel.open(tempFile,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
        }
        try {
            Files.move(tempFile, targetFile, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tempFile, targetFile, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
