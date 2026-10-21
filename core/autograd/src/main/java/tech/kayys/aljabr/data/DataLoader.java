package tech.kayys.aljabr.data;

import java.util.*;
/**
 * 
 * Core class for tech module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public final class DataLoader {
    private final Dataset<Batch> dataset;

    public DataLoader(Dataset<Batch> dataset) {
        this.dataset = dataset;
    }

    public Iterable<Batch> batches() {
        return dataset;
    }
}