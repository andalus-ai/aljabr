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
 *
 * @author bhangun
 */

package tech.kayys.aljabr.spi.model;

import tech.kayys.aljabr.spi.spec.*;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.model.ModelFormat;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
/**
 * Repository for Context data access operations.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record RepositoryContext(
                Path cacheDir,
                Duration timeout,
                Map<String, Object> attributes) {
}