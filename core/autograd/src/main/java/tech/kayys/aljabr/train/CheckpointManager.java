package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;

import java.io.*;
import java.nio.file.*;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
/**
 * Manager/handler for checkpoint operations.
 *
 * <p>Key functionality:
 * <ul>
/**
 * Operations.
 *
 * @author bhangun
 * @since 0.1.0
 */
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class CheckpointManager implements AutoCloseable {

    private final Path checkpointDir;
    private final int maxCheckpoints;
    private final boolean compress;
    private final boolean asyncSave;

    private final PriorityQueue<CheckpointMeta> checkpointQueue;
    private final ExecutorService executor;

    public CheckpointManager(String checkpointDir) {
        this(checkpointDir, 5, true, true);
    }

    public CheckpointManager(String checkpointDir, int maxCheckpoints, boolean compress, boolean asyncSave) {
        this.checkpointDir = Paths.get(checkpointDir);
        this.maxCheckpoints = maxCheckpoints;
        this.compress = compress;
        this.asyncSave = asyncSave;

        // PriorityQueue head is MIN element. To evict the worst (highest loss), we use
        // reversed comparator
        this.checkpointQueue = new PriorityQueue<>(
                Comparator.comparingDouble(CheckpointMeta::validationLoss).reversed());

        this.executor = asyncSave
                ? Executors.newSingleThreadExecutor(r -> {
                    Thread t = new Thread(r, "aljabr-checkpoint-saver");
                    t.setDaemon(true);
                    return t;
                })
                : null;

        try {
            Files.createDirectories(this.checkpointDir);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create checkpoint directory: " + checkpointDir, e);
        }
    }

    public void save(String baseName, Map<String, Tensor> params, double valLoss) {
        String timestamp = DateTimeFormatter.ISO_INSTANT.format(Instant.now());
        String filename = String.format("%s_loss%.4f_%s.bin%s",
                baseName, valLoss, timestamp.replace(":", ""), compress ? ".gz" : "");

        Path path = checkpointDir.resolve(filename);
        CheckpointMeta meta = new CheckpointMeta(valLoss, timestamp, path.toString());

        Runnable saveTask = () -> {
            try {
                writeCheckpoint(path, params);
                synchronized (checkpointQueue) {
                    checkpointQueue.offer(meta);
                    cleanupOldCheckpoints();
                }
            } catch (IOException e) {
                System.err.println("Failed to save checkpoint: " + path);
                e.printStackTrace();
            }
        };

        if (asyncSave && executor != null) {
            executor.submit(saveTask);
        } else {
            saveTask.run();
        }
    }

    // Legacy method for backward compatibility
    public void save(String path, Map<String, Tensor> params) throws IOException {
        writeCheckpoint(Paths.get(path), params);
    }

    private void writeCheckpoint(Path path, Map<String, Tensor> params) throws IOException {
        OutputStream fileOut = Files.newOutputStream(path);
        if (compress) {
            fileOut = new GZIPOutputStream(fileOut);
        }
        try (ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(params);
        }
    }

    private void cleanupOldCheckpoints() {
        while (checkpointQueue.size() > maxCheckpoints) {
            CheckpointMeta evict = checkpointQueue.poll();
            if (evict != null) {
                try {
                    Files.deleteIfExists(Paths.get(evict.path()));
                } catch (IOException e) {
                    System.err.println("Failed to delete evicted checkpoint: " + evict.path());
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Tensor> load(String path) throws IOException, ClassNotFoundException {
        InputStream fileIn = Files.newInputStream(Paths.get(path));
        if (path.endsWith(".gz")) {
            fileIn = new GZIPInputStream(fileIn);
        }
        try (ObjectInputStream in = new ObjectInputStream(fileIn)) {
            return (Map<String, Tensor>) in.readObject();
        }
    }

    @Override
    public void close() {
        if (executor != null) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    public record CheckpointMeta(double validationLoss, String timestamp, String path) {
    }
}