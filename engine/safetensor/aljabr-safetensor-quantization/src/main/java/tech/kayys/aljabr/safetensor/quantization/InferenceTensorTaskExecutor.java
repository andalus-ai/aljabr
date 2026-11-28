/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;

/**
 * Runs independent inference tensor tasks concurrently.
 *
 * <p>The pool is scoped to one invocation and is always shut down before this
 * method returns. Task implementations remain responsible for tensor ownership:
 * captured FFM segments must be accessible from worker threads, and a task must
 * not close an input tensor unless ownership is explicitly transferred by its
 * caller.</p>
 *
 * <p>Task failures are propagated to the caller. Per-tensor fallback behavior
 * belongs in the task itself, where the tensor name and ownership context are
 * available.</p>
 */
final class InferenceTensorTaskExecutor {

    private InferenceTensorTaskExecutor() {
    }

    static <T> void execute(List<T> tasks, Consumer<T> task) {
        Objects.requireNonNull(tasks, "tasks must not be null");
        Objects.requireNonNull(task, "task must not be null");
        if (tasks.isEmpty()) {
            return;
        }

        List<T> snapshot = List.copyOf(tasks);
        int parallelism = Math.max(1,
                Math.min(Runtime.getRuntime().availableProcessors(), snapshot.size()));
        ForkJoinPool pool = new ForkJoinPool(parallelism);
        try {
            pool.submit(() -> snapshot.parallelStream().forEach(task)).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while quantizing inference tensors", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (cause instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            if (cause instanceof Error error) {
                throw error;
            }
            throw new IllegalStateException("Inference tensor task execution failed", cause);
        } finally {
            pool.shutdown();
        }
    }
}
