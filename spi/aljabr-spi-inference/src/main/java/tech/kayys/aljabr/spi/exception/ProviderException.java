/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.exception;

import tech.kayys.aljabr.error.ErrorCode;
import java.util.HashMap;
import java.util.Map;

/**
 * Base exception for provider-related errors.
 */
public class ProviderException extends RuntimeException {

    private final String providerId;
    private final ErrorCode errorCode;
    private final boolean retryable;
    private final Map<String, String> context = new HashMap<>();

    public ProviderException(String message) {
        super(message);
        this.providerId = null;
        this.errorCode = null;
        this.retryable = false;
    }

    public ProviderException(String message, Throwable cause) {
        super(message, cause);
        this.providerId = null;
        this.errorCode = null;
        this.retryable = false;
    }

    public ProviderException(String providerId, String message) {
        super(message);
        this.providerId = providerId;
        this.errorCode = null;
        this.retryable = false;
    }

    public ProviderException(String providerId, String message, Throwable cause) {
        super(message, cause);
        this.providerId = providerId;
        this.errorCode = null;
        this.retryable = false;
    }

    public ProviderException(String providerId, ErrorCode errorCode, String message) {
        super(message);
        this.providerId = providerId;
        this.errorCode = errorCode;
        this.retryable = false;
    }

    public ProviderException(String providerId, ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.providerId = providerId;
        this.errorCode = errorCode;
        this.retryable = false;
    }

    public ProviderException(String providerId, ErrorCode errorCode, String message, boolean retryable) {
        super(message);
        this.providerId = providerId;
        this.errorCode = errorCode;
        this.retryable = retryable;
    }

    public ProviderException(String providerId, ErrorCode errorCode, String message, boolean retryable, Throwable cause) {
        super(message, cause);
        this.providerId = providerId;
        this.errorCode = errorCode;
        this.retryable = retryable;
    }

    public String getProviderId() {
        return providerId;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    public boolean isRetryable() {
        return retryable;
    }

    public Map<String, String> getContext() {
        return context;
    }

    public ProviderException withContext(String key, String value) {
        this.context.put(key, value);
        return this;
    }

    public ProviderException withContext(Map<String, String> context) {
        if (context != null) {
            this.context.putAll(context);
        }
        return this;
    }
}
