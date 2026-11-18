package tech.kayys.aljabr.clustering.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

class ClusteringDatasetTest {

    @Test
    void exposesShapeAndBorrowsTheInputTensor() {
        Tensor vectors = tensorWithShape(new Shape(12, 8), new AtomicBoolean());

        ClusteringDataset dataset = new ClusteringDataset(vectors);

        assertThat(dataset.vectors()).isSameAs(vectors);
        assertThat(dataset.sampleCount()).isEqualTo(12);
        assertThat(dataset.dimensions()).isEqualTo(8);
    }

    @Test
    void doesNotCloseTheBorrowedTensor() {
        AtomicBoolean closed = new AtomicBoolean();
        Tensor vectors = tensorWithShape(new Shape(3, 2), closed);

        new ClusteringDataset(vectors);

        assertThat(closed).isFalse();
    }

    @Test
    void rejectsTensorsThatAreNotRankTwo() {
        assertThatThrownBy(() -> new ClusteringDataset(
                tensorWithShape(new Shape(4), new AtomicBoolean())))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("rank 2");
    }

    @Test
    void rejectsAnEmptySampleDimension() {
        assertThatThrownBy(() -> new ClusteringDataset(
                tensorWithShape(new Shape(0, 4), new AtomicBoolean())))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("at least one sample");
    }

    @Test
    void rejectsAnEmptyVectorDimension() {
        assertThatThrownBy(() -> new ClusteringDataset(
                tensorWithShape(new Shape(4, 0), new AtomicBoolean())))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("at least one dimension");
    }

    private static Tensor tensorWithShape(Shape shape, AtomicBoolean closed) {
        return (Tensor) Proxy.newProxyInstance(
                Tensor.class.getClassLoader(),
                new Class<?>[] { Tensor.class },
                (proxy, method, args) -> {
                    return switch (method.getName()) {
                        case "shape" -> shape;
                        case "close", "release" -> {
                            closed.set(true);
                            yield null;
                        }
                        case "toString" -> "Tensor" + shape;
                        default -> throw new UnsupportedOperationException(
                                "Unexpected tensor operation in dataset contract test: " + method.getName());
                    };
                });
    }
}
