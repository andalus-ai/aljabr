package tech.kayys.aljabr.spi.model.ir;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Unique, model-scoped collection of external asset references. */
public record AssetCollection(List<AssetReference> assets) {
    public AssetCollection {
        assets = List.copyOf(Objects.requireNonNull(assets, "assets must not be null"));
        Set<String> ids = new HashSet<>();
        for (AssetReference asset : assets) {
            Objects.requireNonNull(asset, "asset must not be null");
            if (!ids.add(asset.id())) throw new IllegalArgumentException("duplicate asset id: " + asset.id());
        }
    }

    public static AssetCollection empty() { return new AssetCollection(List.of()); }
}
