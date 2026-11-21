package tech.kayys.aljabr.spi.model.ir;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Unique, model-scoped collection of tokenizer and modality components. */
public record ComponentCollection(List<ModelComponent> components) {
    public ComponentCollection {
        components = List.copyOf(Objects.requireNonNull(components, "components must not be null"));
        Set<String> ids = new HashSet<>();
        for (ModelComponent component : components) {
            Objects.requireNonNull(component, "component must not be null");
            if (!ids.add(component.id())) {
                throw new IllegalArgumentException("duplicate component id: " + component.id());
            }
        }
    }

    public static ComponentCollection empty() { return new ComponentCollection(List.of()); }
}
