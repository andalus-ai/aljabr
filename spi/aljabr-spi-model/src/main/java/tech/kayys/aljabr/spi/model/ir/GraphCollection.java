package tech.kayys.aljabr.spi.model.ir;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Named collection of computation graphs belonging to one model. */
public record GraphCollection(List<GraphDescriptor> graphs) {
    public GraphCollection {
        graphs = List.copyOf(Objects.requireNonNull(graphs, "graphs must not be null"));
        Set<String> ids = new HashSet<>();
        for (GraphDescriptor graph : graphs) {
            Objects.requireNonNull(graph, "graph must not be null");
            if (!ids.add(graph.id())) throw new IllegalArgumentException("duplicate graph id: " + graph.id());
        }
    }

    public static GraphCollection empty() { return new GraphCollection(List.of()); }
}
