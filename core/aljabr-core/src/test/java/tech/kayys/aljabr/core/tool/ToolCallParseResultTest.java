package tech.kayys.aljabr.core.tool;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolCallParseResultTest {
    @Test
    void snapshotsCallsAndDiagnosticsAndReportsPartialResults() {
        var calls = new ArrayList<>(List.of(new ToolCall("lookup", Map.of("id", 7))));
        var diagnostics = new ArrayList<>(List.of(new ToolCallParseDiagnostic(
                "MALFORMED_CALL", ToolCallParseDiagnostic.Severity.ERROR,
                "Malformed call body", 12)));

        var result = new ToolCallParseResult(calls, diagnostics);
        calls.clear();
        diagnostics.clear();

        assertTrue(result.hasCalls());
        assertTrue(result.hasErrors());
        assertFalse(result.isComplete());
        assertFalse(result.isEmpty());
        assertEquals("lookup", result.calls().getFirst().name());
        assertThrows(UnsupportedOperationException.class, () -> result.calls().clear());
        assertThrows(UnsupportedOperationException.class, () -> result.diagnostics().clear());
    }

    @Test
    void emptyCleanResultMeansNoCallsWerePresent() {
        var result = new ToolCallParseResult(List.of(), List.of());
        assertFalse(result.hasCalls());
        assertFalse(result.hasErrors());
        assertTrue(result.isComplete());
        assertTrue(result.isEmpty());
    }

    @Test
    void rejectsInvalidDiagnosticOffsetsAndBlankCodes() {
        assertThrows(IllegalArgumentException.class, () -> new ToolCallParseDiagnostic(
                "BAD", ToolCallParseDiagnostic.Severity.ERROR, "bad offset", -2));
        assertThrows(IllegalArgumentException.class, () -> new ToolCallParseDiagnostic(
                " ", ToolCallParseDiagnostic.Severity.ERROR, "bad code", 0));
    }
}
