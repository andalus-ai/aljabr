/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

package tech.kayys.aljabr.spi.embedding;

import java.util.concurrent.CompletionStage;

/**
 * Model-family execution contract for embedding models.
 *
 * <p>An implementation owns the model-specific forward pass and embedding
 * post-processing. It does not own model discovery, serving admission,
 * scheduling, transport concerns, or request lifecycle policy.
 *
 * <p>Implementations are associated with one canonical model-family identifier.
 * The inference runtime resolves a model identifier to a loaded runner before
 * invoking {@link #embed(EmbeddingRequest)}.
 *
 * <p>The returned stage completes with embeddings in the same order as the
 * request inputs. Implementations must fail the stage when an input or model
 * configuration is unsupported; they must not return placeholder vectors.
 */
public interface EmbeddingModelRunner {

    /**
     * Canonical model-family identifier handled by this runner, for example
     * {@code embedding_gemma2}.
     *
     * @return non-blank model-family identifier
     */
    String modelFamily();

    /**
     * Execute the model-specific embedding computation.
     *
     * @param request validated embedding request
     * @return asynchronous result preserving input order
     */
    CompletionStage<EmbeddingResponse> embed(EmbeddingRequest request);
}
