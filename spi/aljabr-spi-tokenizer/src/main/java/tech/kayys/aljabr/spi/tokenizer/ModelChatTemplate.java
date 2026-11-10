/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import java.util.List;
import tech.kayys.aljabr.spi.Message;

/**
 * SPI for model family chat template formatters.
 *
 * <p>Implemented by model-specific modules (e.g. {@code aljabr-model-gemma4},
 * {@code aljabr-model-llama}, etc.) and discovered via {@link java.util.ServiceLoader}.</p>
 */
public interface ModelChatTemplate {

    /**
     * Checks if this formatter supports the given model architecture / type.
     */
    boolean supports(String modelType);

    /**
     * Formats conversation messages into the model-specific prompt string.
     */
    String format(List<Message> messages);
}
