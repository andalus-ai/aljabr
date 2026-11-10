/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * SafetensorConfigProducer.java
 * ──────────────────────────────
 * CDI producer for SafeTensor configuration.
 */
package tech.kayys.aljabr.safetensor.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;

/**
 * CDI producer for SafeTensor module beans.
 * @author bhangun
 */
@ApplicationScoped
public class SafetensorConfigProducer {

    /**
     * Produce ObjectMapper bean.
     */
    @Produces
    @jakarta.inject.Named("safetensorObjectMapper")
    @jakarta.inject.Singleton
    public ObjectMapper objectMapper() {
        return new ObjectMapper();
    }
}
