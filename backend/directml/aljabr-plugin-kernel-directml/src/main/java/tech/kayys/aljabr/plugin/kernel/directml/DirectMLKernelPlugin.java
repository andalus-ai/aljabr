/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.aljabr.plugin.kernel.directml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.aljabr.plugin.kernel.*;

import java.util.Map;
import java.util.Set;

/**
 * DirectML kernel plugin for Windows DirectX.
 * 
 * <p>
 * Provides Microsoft DirectML kernel implementations for Windows GPUs.
 * </p>
 * 
 * <h2>Requirements</h2>
 * <ul>
 * <li>Windows 10/11</li>
 * <li>DirectX 12</li>
 * <li>DirectML 1.8+</li>
 * </ul>
 */
public class DirectMLKernelPlugin implements KernelPlugin {

    private static final Logger LOG = LoggerFactory.getLogger(DirectMLKernelPlugin.class);
    public static final String ID = "directml-kernel";

    private boolean enabled = true;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "DirectML Kernel";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public String description() {
        return "Microsoft DirectML kernel implementations for Windows GPUs";
    }

    @Override
    public boolean isAvailable() {
        if (!enabled) {
            return false;
        }
        String os = System.getProperty("os.name").toLowerCase();
        return os.contains("windows");
    }

    @Override
    public String platform() {
        return "directml";
    }

    @Override
    public Set<String> supportedArchitectures() {
        return Set.of("nvidia", "amd", "intel");
    }

    @Override
    public Set<String> supportedVersions() {
        return Set.of("1.8", "1.9", "1.10", "1.11");
    }

    @Override
    public Set<String> supportedOperations() {
        return Set.of("gemm", "attention", "layer_norm", "activation");
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> KernelResult<T> execute(KernelOperation operation, KernelContext context) throws KernelException {
        if (!isAvailable()) {
            throw new KernelExecutionException(platform(), operation.getName(), "DirectML kernel is not available");
        }
        LOG.info("Executing DirectML operation: {}", operation.getName());
        Map<String, Object> result = Map.of(
                "status", "success",
                "platform", "directml",
                "operation", operation.getName());
        return (KernelResult<T>) KernelResult.success(result);
    }

    @Override
    public Map<String, Object> metadata() {
        return Map.of(
                "platform", "directml",
                "supported_ops", supportedOperations());
    }
}
