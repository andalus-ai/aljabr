/*
 * Aljabr Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.safetensor.quantization;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.regex.Pattern;

/**
 * Explicit registry of quantized tensor formats that this process knows how to validate.
 *
 * <p>Registration is an assertion that a consumer-side implementation understands the
 * registered format version. Merely constructing a {@link QuantizedTensorDescriptor}
 * does not register or enable a format. This registry intentionally does not provide
 * codecs, quantizers, or execution kernels.</p>
 *
 * <p>Registries are instance-scoped so applications can control which formats are
 * admitted. No formats are enabled by default.</p>
 */
public final class QuantizationFormatRegistry {
    private static final Pattern FORMAT_ID =
            Pattern.compile("[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,127}");

    private final ConcurrentMap<FormatKey, FormatDefinition> definitions =
            new ConcurrentHashMap<>();

    /** Register a format/version pair. Duplicate registrations are rejected. */
    public void register(FormatDefinition definition) {
        Objects.requireNonNull(definition, "definition");
        FormatDefinition existing = definitions.putIfAbsent(definition.key(), definition);
        if (existing != null) {
            throw new IllegalStateException("Quantization format is already registered: "
                    + definition.key());
        }
    }

    /** Find an exact format/version registration. No version fallback is performed. */
    public Optional<FormatDefinition> find(String formatId, int formatVersion) {
        return Optional.ofNullable(definitions.get(new FormatKey(formatId, formatVersion)));
    }

    /** Return whether the exact format/version pair is registered. */
    public boolean supports(String formatId, int formatVersion) {
        return definitions.containsKey(new FormatKey(formatId, formatVersion));
    }

    /**
     * Validate a descriptor against its exact registered format version.
     *
     * @throws UnsupportedQuantizationFormatException if no exact registration exists
     * @throws IllegalArgumentException if the registered validator rejects the descriptor
     */
    public void validate(QuantizedTensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        FormatKey key = new FormatKey(descriptor.getFormatId(), descriptor.getFormatVersion());
        FormatDefinition definition = definitions.get(key);
        if (definition == null) {
            throw new UnsupportedQuantizationFormatException(
                    "Unsupported quantization format/version: " + key);
        }
        definition.validate(descriptor);
    }

    /** Return an immutable, key-sorted snapshot of registered definitions. */
    public Map<FormatKey, FormatDefinition> getDefinitions() {
        return Collections.unmodifiableMap(new TreeMap<>(definitions));
    }

    /** Exact identity of a serialized quantization format version. */
    public record FormatKey(String formatId, int formatVersion)
            implements Comparable<FormatKey> {
        public FormatKey {
            if (formatId == null || !FORMAT_ID.matcher(formatId).matches()) {
                throw new IllegalArgumentException("formatId must be a stable identifier (max 128 characters)");
            }
            if (formatVersion < 1) {
                throw new IllegalArgumentException("formatVersion must be >= 1");
            }
        }

        @Override
        public int compareTo(FormatKey other) {
            int idOrder = formatId.compareTo(other.formatId);
            return idOrder != 0 ? idOrder : Integer.compare(formatVersion, other.formatVersion);
        }

        @Override
        public String toString() {
            return formatId + "@" + formatVersion;
        }
    }

    /** Validation contract for one exact serialized format version. */
    @FunctionalInterface
    public interface DescriptorValidator {
        void validate(QuantizedTensorDescriptor descriptor);
    }

    /** Immutable registration of a format/version and its structural validation rules. */
    public static final class FormatDefinition {
        private final FormatKey key;
        private final String description;
        private final DescriptorValidator validator;

        public FormatDefinition(
                String formatId,
                int formatVersion,
                String description,
                DescriptorValidator validator) {
            this.key = new FormatKey(formatId, formatVersion);
            if (description == null || description.isBlank()) {
                throw new IllegalArgumentException("description must not be blank");
            }
            this.description = description;
            this.validator = Objects.requireNonNull(validator, "validator");
        }

        public FormatKey key() {
            return key;
        }

        public String formatId() {
            return key.formatId();
        }

        public int formatVersion() {
            return key.formatVersion();
        }

        public String description() {
            return description;
        }

        private void validate(QuantizedTensorDescriptor descriptor) {
            if (!key.formatId().equals(descriptor.getFormatId())
                    || key.formatVersion() != descriptor.getFormatVersion()) {
                throw new IllegalArgumentException("Descriptor identity does not match registration " + key);
            }
            validator.validate(descriptor);
        }
    }

    /** Raised when a descriptor names a format/version that is not explicitly enabled. */
    public static final class UnsupportedQuantizationFormatException
            extends IllegalArgumentException {
        public UnsupportedQuantizationFormatException(String message) {
            super(message);
        }
    }
}
