package tech.kayys.aljabr.backend.cpu;

/**
 * Visits the elements selected along one tensor axis in logical row-major order.
 *
 * <p>This helper owns only the common outer/selected/inner traversal. Callers
 * retain responsibility for validation, source layout access, destination
 * addressing, and the operation-specific element transfer.</p>
 */
final class CpuIndexedAxisTraversal {

    private CpuIndexedAxisTraversal() {
    }

    static void forEach(
            long outerSize,
            long sourceAxisSize,
            long innerSize,
            int[] selected,
            ElementConsumer consumer) {
        if (outerSize < 0 || sourceAxisSize < 0 || innerSize < 0) {
            throw new IllegalArgumentException("Indexed axis dimensions must not be negative");
        }
        if (selected == null) {
            throw new NullPointerException("selected");
        }
        if (consumer == null) {
            throw new NullPointerException("consumer");
        }

        for (long outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (int selectionPosition = 0; selectionPosition < selected.length; selectionPosition++) {
                int selectedIndex = selected[selectionPosition];
                long sourceBase = Math.multiplyExact(
                        Math.addExact(Math.multiplyExact(outerIndex, sourceAxisSize), selectedIndex),
                        innerSize);
                for (long innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                    long sourceLogicalIndex = Math.addExact(sourceBase, innerIndex);
                    long outputLogicalIndex = Math.addExact(
                            Math.multiplyExact(
                                    Math.addExact(Math.multiplyExact(outerIndex, selected.length), selectionPosition),
                                    innerSize),
                            innerIndex);
                    consumer.accept(outerIndex, selectionPosition, selectedIndex,
                            innerIndex, sourceLogicalIndex, outputLogicalIndex);
                }
            }
        }
    }

    @FunctionalInterface
    interface ElementConsumer {
        void accept(long outerIndex, int selectionPosition, int selectedIndex,
                long innerIndex, long sourceLogicalIndex, long outputLogicalIndex);
    }
}
