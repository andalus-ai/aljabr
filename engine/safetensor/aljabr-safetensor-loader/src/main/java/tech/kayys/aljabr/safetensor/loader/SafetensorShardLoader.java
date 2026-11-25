/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorShardLoader.java
 * ──────────────────────────
 * Unified entry-point that auto-detects whether a model is single-file or
 * multi-shard, and provides a consistent tensor-access API for both cases.
 *
 * Decision logic
 * ══════════════
 *   Path is a file                 → load as single SafeTensors file
 *   Path is a directory + index    → load as multi-shard model
 *   Path is a directory, no index  → scan for a single .safetensors file
 *
 * Multi-shard tensor access
 * ═════════════════════════
 * Shards are opened lazily — only the shard files that contain actually
 * requested tensors are memory-mapped.  This is crucial for large models
 * where you may only need a subset of weights (e.g. for LoRA adapter merging).
 *
 * Each open shard is kept in a cache and reused for subsequent lookups
 * within the same session.  The cache is evicted on {@link #close()}.
 */
package tech.kayys.aljabr.safetensor.loader;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.safetensor.exception.SafetensorException;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Auto-detecting, lazy-loading SafeTensors loader for both single-file and
 * multi-shard model checkpoints.
 *
 * <p>
 * This is the recommended CDI entry point for production use — it handles
 * all file/directory layouts used by HuggingFace Hub-style model repositories.
 *
 * <p>
 * <b>Usage:</b>
 * 
 * <pre>{@code
 * @Inject
 * SafetensorShardLoader loader;
 *
 * try (SafetensorShardSession session = loader.open(Path.of("/models/llama2-70b"))) {
 *     SafetensorTensor q = session.tensor("model.layers.0.self_attn.q_proj.weight");
 *     float[] weights = q.toFloatArray();
 * }
 * }</pre>
 * @author bhangun
 */
@ApplicationScoped
public class SafetensorShardLoader {

    private static final Logger log = Logger.getLogger(SafetensorShardLoader.class);

    @Inject
    SafetensorFFMLoader ffmLoader;

    @Inject
    ObjectMapper objectMapper;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Open a SafeTensors model checkpoint at the given path.
     *
     * <p>
     * The path may be:
     * <ul>
     * <li>A {@code .safetensors} file — single-file model
     * <li>A directory containing a {@code model.safetensors.index.json}
     * — multi-shard model (shards loaded lazily)
     * <li>A directory without an index — single-file model found by scanning
     * </ul>
     *
     * @param modelPath path to the model file or directory
     * @return an open {@link SafetensorShardSession}
     * @throws SafetensorException on detection or loading errors
     */
    public SafetensorShardSession open(Path modelPath) {
        Objects.requireNonNull(modelPath, "modelPath must not be null");
        Path resolved = modelPath.toAbsolutePath().normalize();

        if (Files.isRegularFile(resolved)) {
            // Direct single-file load
            log.infof("SafetensorShardLoader: opening single-file model [%s]", resolved);
            SafetensorLoadResult result = ffmLoader.load(resolved);
            return SafetensorShardSession.singleFile(result);
        }

        if (Files.isDirectory(resolved)) {
            if (SafetensorShardIndex.isShardedModel(resolved)) {
                // Multi-shard model
                log.infof("SafetensorShardLoader: opening multi-shard model [%s]", resolved);
                Path indexPath = SafetensorShardIndex.resolveIndexPath(resolved);
                SafetensorShardIndex index = SafetensorShardIndex.load(indexPath, objectMapper);
                return SafetensorShardSession.sharded(index, ffmLoader);
            } else {
                // Scan for a single safetensors file in the directory
                Path candidate = findSingleSafetensorFile(resolved);
                log.infof("SafetensorShardLoader: found single file in dir [%s]", candidate);
                SafetensorLoadResult result = ffmLoader.load(candidate);
                return SafetensorShardSession.singleFile(result);
            }
        }

        throw new SafetensorException.IoException(
                "Path does not exist or is not a file/directory: " + resolved, resolved, null);
    }

    /**
     * Inspect the header(s) of a model without loading tensor data.
     * Returns a merged view of all tensor metadata across all shards.
     *
     * @param modelPath path to the model file or directory
     * @return list of all tensor info objects (across all shards)
     */
    public List<SafetensorTensorInfo> inspectTensors(Path modelPath) {
        Objects.requireNonNull(modelPath);
        Path resolved = modelPath.toAbsolutePath().normalize();

        if (Files.isRegularFile(resolved)) {
            SafetensorHeader hdr = ffmLoader.loadHeaderOnly(resolved);
            return new ArrayList<>(hdr.tensors().values());
        }

        if (Files.isDirectory(resolved)) {
            if (SafetensorShardIndex.isShardedModel(resolved)) {
                Path indexPath = SafetensorShardIndex.resolveIndexPath(resolved);
                SafetensorShardIndex index = SafetensorShardIndex.load(indexPath, objectMapper);

                Map<String, SafetensorHeader> headers = loadShardHeaders(index);
                index.validateShardHeaders(headers);

                List<SafetensorTensorInfo> all = new ArrayList<>();
                for (String shardName : index.shardFileNames()) {
                    all.addAll(headers.get(shardName).tensors().values());
                }
                return Collections.unmodifiableList(all);
            } else {
                Path candidate = findSingleSafetensorFile(resolved);
                SafetensorHeader hdr = ffmLoader.loadHeaderOnly(candidate);
                return new ArrayList<>(hdr.tensors().values());
            }
        }

        throw new SafetensorException.IoException(
                "Path not found: " + resolved, resolved, null);
    }

    /**
     * Validate a sharded model's index against every shard header without
     * materializing tensor payloads. This is an explicit integrity check so
     * callers can keep {@link #open(Path)} lazy for inference startup.
     *
     * @param modelDir directory containing the index and shard files
     * @throws SafetensorException if the path is not a sharded model or the
     *         index disagrees with shard headers
     */
    public void validateIndexConsistency(Path modelDir) {
        Objects.requireNonNull(modelDir, "modelDir must not be null");
        Path resolved = modelDir.toAbsolutePath().normalize();
        if (!Files.isDirectory(resolved) || !SafetensorShardIndex.isShardedModel(resolved)) {
            throw new SafetensorException.ValidationException(
                    "Index consistency validation requires a directory containing a shard index",
                    resolved);
        }
        SafetensorShardIndex index = SafetensorShardIndex.load(
                SafetensorShardIndex.resolveIndexPath(resolved), objectMapper);
        index.validateShardHeaders(loadShardHeaders(index));
    }

    /**
     * Visit every tensor in a checkpoint while retaining at most one shard's
     * mapped storage at a time.
     *
     * <p>Unlike {@link SafetensorShardSession}, this traversal API does not
     * retain opened shard mappings after each callback. It is intended for
     * conversion, validation, indexing, and other sequential workloads that
     * process every tensor without keeping the complete model resident.
     *
     * <p>The supplied tensor view is valid only for the duration of the
     * callback. Consumers must not retain the view or its {@code MemorySegment}
     * after the callback returns. Copy data explicitly if it must outlive the
     * callback.
     *
     * @param modelPath model file or directory containing a shard index
     * @param consumer receives each tensor name and its zero-copy view
     */
    public void forEachTensor(Path modelPath, TensorConsumer consumer) {
        Objects.requireNonNull(modelPath, "modelPath must not be null");
        Objects.requireNonNull(consumer, "consumer must not be null");
        Path resolved = modelPath.toAbsolutePath().normalize();

        if (Files.isRegularFile(resolved)) {
            visitFile(resolved, consumer, null);
            return;
        }
        if (!Files.isDirectory(resolved)) {
            throw new SafetensorException.IoException(
                    "Path does not exist or is not a file/directory: " + resolved,
                    resolved, null);
        }

        if (!SafetensorShardIndex.isShardedModel(resolved)) {
            visitFile(findSingleSafetensorFile(resolved), consumer, null);
            return;
        }

        SafetensorShardIndex index = SafetensorShardIndex.load(
                SafetensorShardIndex.resolveIndexPath(resolved), objectMapper);
        for (String shardName : index.shardFileNames()) {
            visitFile(index.shardPath(shardName), consumer, index.tensorsInShard(shardName));
        }
    }

    private void visitFile(Path shardPath, TensorConsumer consumer, List<String> selectedNames) {
        try (SafetensorLoadResult result = ffmLoader.load(shardPath)) {
            Collection<String> names = selectedNames == null
                    ? result.tensorNames()
                    : selectedNames;
            for (String name : names) {
                SafetensorTensor tensor = result.tensor(name);
                consumer.accept(name, tensor);
            }
        }
    }

    /**
     * Callback for sequential tensor traversal. The tensor view and its backing
     * memory must not be retained after {@link #accept} returns.
     */
    @FunctionalInterface
    public interface TensorConsumer {
        void accept(String name, SafetensorTensor tensor);
    }

    private Map<String, SafetensorHeader> loadShardHeaders(SafetensorShardIndex index) {
        Map<String, SafetensorHeader> headers = new LinkedHashMap<>();
        for (String shardName : index.shardFileNames()) {
            headers.put(shardName, ffmLoader.loadHeaderOnly(index.shardPath(shardName)));
        }
        return headers;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    private Path findSingleSafetensorFile(Path dir) {
        try (var stream = Files.walk(dir, 2)) {
            List<Path> candidates = stream
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return n.endsWith(".safetensors") || n.endsWith(".safetensor");
                    })
                    .collect(Collectors.toList());

            if (candidates.isEmpty()) {
                throw new SafetensorException.IoException(
                        "No .safetensors file found in directory: " + dir, dir, null);
            }
            if (candidates.size() > 1) {
                log.warnf("Multiple .safetensors files found in [%s], using first: %s",
                        dir, candidates.get(0));
            }
            return candidates.get(0);
        } catch (SafetensorException e) {
            throw e;
        } catch (IOException e) {
            throw new SafetensorException.IoException(
                    "Error scanning directory for .safetensors files", dir, e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inner class: Session
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * An open model session that provides tensor access.
     *
     * <p>
     * For multi-shard models, shard files are opened (mmap'd) lazily on
     * first tensor access and cached for the session lifetime.
     */
    public static final class SafetensorShardSession implements AutoCloseable {

        /** {@code true} for single-file models. */
        private final boolean singleFile;

        /** Single-file result (non-null only for single-file models). */
        private final SafetensorLoadResult singleResult;

        /** Multi-shard index (non-null only for sharded models). */
        private final SafetensorShardIndex shardIndex;

        /** Lazily-opened shard results, keyed by shard filename. */
        private final Map<String, SafetensorLoadResult> openShards;

        /** FFM loader used to open additional shards on demand. */
        private final SafetensorFFMLoader ffmLoader;

        /** Guards public session access after the caller closes its owner reference. */
        private volatile boolean ownerClosed = false;

        /** True only after all owners and retained leases have been released. */
        private boolean resourcesClosed = false;

        /** One initial reference belongs to the session creator. */
        private int references = 1;

        // Single-file constructor
        private SafetensorShardSession(SafetensorLoadResult single) {
            this.singleFile = true;
            this.singleResult = single;
            this.shardIndex = null;
            this.openShards = null;
            this.ffmLoader = null;
        }

        // Sharded constructor
        private SafetensorShardSession(SafetensorShardIndex index, SafetensorFFMLoader loader) {
            this.singleFile = false;
            this.singleResult = null;
            this.shardIndex = index;
            this.openShards = new ConcurrentHashMap<>();
            this.ffmLoader = loader;
        }

        static SafetensorShardSession singleFile(SafetensorLoadResult result) {
            return new SafetensorShardSession(result);
        }

        static SafetensorShardSession sharded(SafetensorShardIndex index, SafetensorFFMLoader loader) {
            return new SafetensorShardSession(index, loader);
        }

        /**
         * Retain the mapped shard resources beyond this session owner's lifetime.
         *
         * <p>The returned lease must be closed exactly once. It is intended for
         * zero-copy consumers that capture tensor segments and need those
         * segments to remain valid after the session owner closes. The lease
         * does not permit further lookups through this session.</p>
         *
         * @return an independently closeable resource-lifetime lease
         * @throws IllegalStateException if the session owner has already closed
         */
        public synchronized SessionLease retain() {
            if (ownerClosed || resourcesClosed) {
                throw new IllegalStateException("SafetensorShardSession is closed.");
            }
            references = Math.addExact(references, 1);
            return new SessionLease(this);
        }

        /** Whether the underlying mapped resources have been physically released. */
        public synchronized boolean resourcesClosed() {
            return resourcesClosed;
        }

        // ── AccelTensor access ─────────────────────────────────────────────────────

        /**
         * Get a tensor by name.
         *
         * @param name tensor key (e.g. {@code "model.layers.31.mlp.gate_proj.weight"})
         * @return the tensor view
         * @throws SafetensorException.TensorNotFoundException if not found
         * @throws IllegalStateException                       if the session is closed
         */
        public SafetensorTensor tensor(String name) {
            checkOpen();
            if (singleFile) {
                return singleResult.tensor(name);
            }
            // Lazy shard open
            Path shardPath = shardIndex.shardPathForTensor(name);
            String shardName = shardPath.getFileName().toString();
            SafetensorLoadResult shard = openShards.computeIfAbsent(
                    shardName, k -> ffmLoader.load(shardPath));
            return shard.tensor(name);
        }

        /** Find a tensor by name, returning empty if absent. */
        public Optional<SafetensorTensor> findTensor(String name) {
            checkOpen();
            if (singleFile) {
                return singleResult.findTensor(name);
            }
            if (!shardIndex.hasTensor(name))
                return Optional.empty();
            return Optional.of(tensor(name));
        }

        /** Whether this session is a sharded model. */
        public boolean isSharded() {
            return !singleFile;
        }

        /** Number of tensors (across all shards). */
        public int tensorCount() {
            return singleFile ? singleResult.tensorCount() : shardIndex.tensorCount();
        }

        /** All tensor names in this session. */
        public Set<String> tensorNames() {
            checkOpen();
            return singleFile
                    ? singleResult.tensorNames()
                    : shardIndex.weightMap().keySet();
        }

        // ── Lifecycle ─────────────────────────────────────────────────────────

        @Override
        public void close() {
            synchronized (this) {
                if (ownerClosed) {
                    return;
                }
                ownerClosed = true;
            }
            releaseReference();
        }

        private void releaseReference() {
            boolean closeResources;
            synchronized (this) {
                if (references <= 0) {
                    throw new IllegalStateException("SafetensorShardSession reference count underflow.");
                }
                references--;
                closeResources = references == 0 && !resourcesClosed;
                if (closeResources) {
                    resourcesClosed = true;
                }
            }
            if (closeResources) {
                closeMappedResources();
            }
        }

        private void closeMappedResources() {
            if (singleFile) {
                safeClose(singleResult);
            } else {
                openShards.values().forEach(SafetensorShardSession::safeClose);
                openShards.clear();
            }
        }

        private void checkOpen() {
            if (ownerClosed || resourcesClosed) {
                throw new IllegalStateException("SafetensorShardSession is closed.");
            }
        }

        /**
         * A retained reference to the session's mapped resources.
         *
         * <p>Closing the original session owner does not invalidate mapped
         * tensors while a lease remains open. The final lease release closes
         * all single-file or cached shard results.</p>
         */
        public static final class SessionLease implements AutoCloseable {
            private final SafetensorShardSession session;
            private final java.util.concurrent.atomic.AtomicBoolean released =
                    new java.util.concurrent.atomic.AtomicBoolean();

            private SessionLease(SafetensorShardSession session) {
                this.session = session;
            }

            @Override
            public void close() {
                if (released.compareAndSet(false, true)) {
                    session.releaseReference();
                }
            }

            public boolean isClosed() {
                return released.get();
            }
        }

        private static void safeClose(AutoCloseable c) {
            try {
                if (c != null)
                    c.close();
            } catch (Exception e) {
                /* best effort */ }
        }

        @Override
        public String toString() {
            return "SafetensorShardSession{"
                    + (singleFile ? "file=" + singleResult.filePath()
                            : "shards=" + shardIndex.shardCount()
                                    + ", tensors=" + shardIndex.tensorCount())
                    + ", ownerClosed=" + ownerClosed
                    + ", resourcesClosed=" + resourcesClosed
                    + ", references=" + references
                    + '}';
        }
    }
}
