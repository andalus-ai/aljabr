package tech.kayys.aljabr.gguf.api;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * Deterministic registry for GGUF model-family adapters.
 *
 * <p>The registry performs artifact inspection only. It never selects an
 * execution provider and never interprets a family match as runtime support.</p>
 */
public final class GgufModelFamilyRegistry {

    private final List<GgufModelFamilyAdapter> adapters;

    public GgufModelFamilyRegistry(Collection<? extends GgufModelFamilyAdapter> adapters) {
        Objects.requireNonNull(adapters, "adapters");
        List<GgufModelFamilyAdapter> ordered = new ArrayList<>();
        Set<String> familyIds = new HashSet<>();
        for (GgufModelFamilyAdapter adapter : adapters) {
            Objects.requireNonNull(adapter, "adapter");
            String familyId = requireText(adapter.familyId(), "adapter familyId");
            if (!familyIds.add(familyId)) {
                throw new IllegalArgumentException("Duplicate GGUF model-family adapter: " + familyId);
            }
            ordered.add(adapter);
        }
        ordered.sort(Comparator.comparing(GgufModelFamilyAdapter::familyId));
        this.adapters = List.copyOf(ordered);
    }

    /** Discovers adapters registered through Java's ServiceLoader mechanism. */
    public static GgufModelFamilyRegistry discover(ClassLoader classLoader) {
        Objects.requireNonNull(classLoader, "classLoader");
        List<GgufModelFamilyAdapter> discovered = ServiceLoader
                .load(GgufModelFamilyAdapter.class, classLoader)
                .stream()
                .map(ServiceLoader.Provider::get)
                .toList();
        return new GgufModelFamilyRegistry(discovered);
    }

    /** Returns the registered adapters in stable family-id order. */
    public List<GgufModelFamilyAdapter> adapters() {
        return adapters;
    }

    /**
     * Inspects an artifact against every registered adapter.
     *
     * <p>A unique match is returned only if every adapter completed inspection.
     * An adapter failure prevents a potentially unsafe unique-family decision.</p>
     */
    public Resolution resolve(GgufModelDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        List<GgufModelFamilyInspection> inspections = new ArrayList<>();
        List<String> failures = new ArrayList<>();
        List<GgufModelFamilyAdapter> matches = new ArrayList<>();

        for (GgufModelFamilyAdapter adapter : adapters) {
            String familyId = adapter.familyId();
            try {
                GgufModelFamilyInspection inspection = Objects.requireNonNull(
                        adapter.inspect(descriptor),
                        "adapter returned null inspection");
                if (!familyId.equals(inspection.familyId())) {
                    throw new IllegalStateException("Adapter " + familyId
                            + " returned inspection for " + inspection.familyId());
                }
                inspections.add(inspection);
                if (inspection.matched()) {
                    matches.add(adapter);
                }
            } catch (RuntimeException exception) {
                String detail = exception.getMessage();
                failures.add(familyId + ": " + exception.getClass().getSimpleName()
                        + (detail == null || detail.isBlank() ? "" : " — " + detail));
            }
        }

        if (!failures.isEmpty()) {
            return new Resolution(Status.ADAPTER_FAILURE, Optional.empty(), inspections, failures);
        }
        if (matches.size() > 1) {
            return new Resolution(Status.AMBIGUOUS, Optional.empty(), inspections,
                    List.of("Multiple model-family adapters matched: "
                            + matches.stream().map(GgufModelFamilyAdapter::familyId).toList()));
        }
        if (matches.size() == 1) {
            return new Resolution(Status.MATCH, Optional.of(matches.getFirst()), inspections, List.of());
        }
        return new Resolution(Status.NO_MATCH, Optional.empty(), inspections, List.of());
    }

    public enum Status {
        MATCH,
        NO_MATCH,
        AMBIGUOUS,
        ADAPTER_FAILURE
    }

    public record Resolution(
            Status status,
            Optional<GgufModelFamilyAdapter> adapter,
            List<GgufModelFamilyInspection> inspections,
            List<String> diagnostics) {

        public Resolution {
            Objects.requireNonNull(status, "status");
            adapter = Objects.requireNonNull(adapter, "adapter");
            inspections = List.copyOf(Objects.requireNonNull(inspections, "inspections"));
            diagnostics = List.copyOf(Objects.requireNonNull(diagnostics, "diagnostics"));
            if ((status == Status.MATCH) != adapter.isPresent()) {
                throw new IllegalArgumentException("Only MATCH resolutions may contain an adapter");
            }
            if ((status == Status.ADAPTER_FAILURE || status == Status.AMBIGUOUS)
                    && diagnostics.isEmpty()) {
                throw new IllegalArgumentException(status + " requires diagnostics");
            }
            if ((status == Status.MATCH || status == Status.NO_MATCH) && !diagnostics.isEmpty()) {
                throw new IllegalArgumentException(status + " cannot contain failure diagnostics");
            }
        }

        public boolean matched() {
            return status == Status.MATCH;
        }
    }

    private static String requireText(String value, String name) {
        Objects.requireNonNull(value, name);
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return normalized;
    }
}
