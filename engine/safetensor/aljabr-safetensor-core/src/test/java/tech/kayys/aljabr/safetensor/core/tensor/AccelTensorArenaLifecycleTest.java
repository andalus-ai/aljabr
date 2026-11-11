/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.core.tensor;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.MemorySegment;

import org.junit.jupiter.api.Test;

class AccelTensorArenaLifecycleTest {

    @Test
    void closeReleasesOwnedTensorStorage() {
        AccelTensor tensor = AccelTensor.zeros(4);
        MemorySegment storage = tensor.dataPtr();

        assertTrue(storage.scope().isAlive());

        tensor.close();

        assertFalse(storage.scope().isAlive());
    }

    @Test
    void closeReleasesMaterializedContiguousCopy() {
        AccelTensor source = AccelTensor.zeros(2, 3);
        AccelTensor transposed = source.transpose(0, 1);
        AccelTensor contiguous = transposed.contiguous();
        MemorySegment storage = contiguous.dataPtr();

        assertTrue(storage.scope().isAlive());

        contiguous.close();

        assertFalse(storage.scope().isAlive());

        transposed.close();
        source.close();
    }
}
