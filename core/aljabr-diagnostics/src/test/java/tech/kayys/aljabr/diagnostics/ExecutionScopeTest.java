package tech.kayys.aljabr.diagnostics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class ExecutionScopeTest {

    @Test
    void emitsStartedAndSucceededWithMonotonicDuration() {
        List<ExecutionEvent> events = new ArrayList<>();
        DiagnosticContext context = context();

        try (ExecutionScope scope = ExecutionScope.start(context, events::add)) {
            assertFalse(scope.isCompleted());
            scope.succeed();
            assertTrue(scope.isCompleted());
        }

        assertEquals(2, events.size());
        assertEquals(ExecutionEvent.Type.STARTED, events.get(0).type());
        assertEquals(0, events.get(0).elapsedNanos());
        assertEquals(ExecutionEvent.Type.SUCCEEDED, events.get(1).type());
        assertTrue(events.get(1).elapsedNanos() >= 0);
        assertEquals(context, events.get(1).context());
    }

    @Test
    void emitsFailureTypeWithoutExceptionMessage() {
        List<ExecutionEvent> events = new ArrayList<>();
        RuntimeException failure = new IllegalStateException("sensitive prompt content");

        try (ExecutionScope scope = ExecutionScope.start(context(), events::add)) {
            scope.fail(failure);
        }

        assertEquals(2, events.size());
        ExecutionEvent failed = events.get(1);
        assertEquals(ExecutionEvent.Type.FAILED, failed.type());
        assertEquals(IllegalStateException.class.getName(), failed.failureType());
        assertFalse(failed.toString().contains("sensitive prompt content"));
    }

    @Test
    void closingAnUnfinishedScopeEmitsAbandoned() {
        List<ExecutionEvent> events = new ArrayList<>();

        try (ExecutionScope ignored = ExecutionScope.start(context(), events::add)) {
            // Deliberately leave the outcome unspecified.
        }

        assertEquals(2, events.size());
        assertEquals(ExecutionEvent.Type.ABANDONED, events.get(1).type());
    }

    @Test
    void observerRuntimeFailureDoesNotReplaceComputationOutcome() {
        RuntimeException observerProblem = new IllegalArgumentException("observer unavailable");

        try (ExecutionScope scope = ExecutionScope.start(context(), event -> {
            throw observerProblem;
        })) {
            int result = 6 * 7;
            assertEquals(42, result);
            assertTrue(scope.observerFailure().isPresent());
            assertEquals(observerProblem, scope.observerFailure().orElseThrow());
            scope.succeed();
        }
    }

    @Test
    void requiresExplicitCorrelationAndOperation() {
        assertThrows(IllegalArgumentException.class,
                () -> ExecutionScope.start(DiagnosticContext.empty()));
        assertThrows(IllegalArgumentException.class, () -> ExecutionScope.start(
                DiagnosticContext.builder().correlationId("trace-1").build()));
    }

    @Test
    void rejectsMultipleExplicitTerminalTransitions() {
        try (ExecutionScope scope = ExecutionScope.start(context())) {
            scope.succeed();
            assertThrows(IllegalStateException.class, scope::succeed);
            assertThrows(IllegalStateException.class,
                    () -> scope.fail(new IllegalStateException()));
        }
    }

    @Test
    void failedEventRequiresFailureType() {
        assertThrows(IllegalArgumentException.class, () -> new ExecutionEvent(
                ExecutionEvent.Type.FAILED,
                context(),
                java.time.Instant.now(),
                1,
                null));
        assertNotNull(ExecutionObserver.NOOP);
        assertEquals(ExecutionObserver.NOOP, ExecutionObserver.noop());
    }

    private static DiagnosticContext context() {
        return DiagnosticContext.builder()
                .correlationId("trace-1")
                .operation("tensor.matmul")
                .component("tensor")
                .provider("cpu")
                .build();
    }
}
