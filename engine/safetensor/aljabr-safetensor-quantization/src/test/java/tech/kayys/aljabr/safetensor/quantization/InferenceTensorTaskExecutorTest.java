/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class InferenceTensorTaskExecutorTest {

    @Test
    void executesEveryTaskBeforeReturning() {
        AtomicInteger completed = new AtomicInteger();

        InferenceTensorTaskExecutor.execute(List.of(1, 2, 3, 4, 5), ignored -> completed.incrementAndGet());

        assertEquals(5, completed.get());
    }

    @Test
    void emptyTaskListIsANoOp() {
        assertDoesNotThrow(() -> InferenceTensorTaskExecutor.execute(List.of(), ignored -> {
            throw new AssertionError("No task should be invoked");
        }));
    }

    @Test
    void propagatesTaskFailuresToTheCaller() {
        IllegalStateException failure = new IllegalStateException("quantizer failed");

        IllegalStateException thrown = assertThrows(IllegalStateException.class,
                () -> InferenceTensorTaskExecutor.execute(List.of(1, 2, 3), value -> {
                    if (value == 2) {
                        throw failure;
                    }
                }));

        assertSame(failure, thrown);
    }
}
