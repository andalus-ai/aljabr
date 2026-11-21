package tech.kayys.aljabr.models.functiongemma;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaToolCallParserTest {
    private final FunctionGemmaToolCallParser parser = new FunctionGemmaToolCallParser();

    @Test
    void parsesEscapedStringsAndPrimitiveArguments() {
        var calls = parser.parse(
                "<start_function_call>call:send_email{"
                + "to:<escape>dev@example.com<escape>,"
                + "subject:<escape>Q4, {review}<escape>,"
                + "urgent:true,retries:3,body:None"
                + "}<end_function_call>");

        assertEquals(1, calls.size());
        var call = calls.getFirst();
        assertEquals("send_email", call.name());
        assertEquals("dev@example.com", call.arguments().get("to"));
        assertEquals("Q4, {review}", call.arguments().get("subject"));
        assertEquals(Boolean.TRUE, call.arguments().get("urgent"));
        assertEquals(3L, call.arguments().get("retries"));
        assertTrue(call.arguments().containsKey("body"));
        assertNull(call.arguments().get("body"));
    }

    @Test
    void parsesMultipleCallsAndNestedArgumentStructures() {
        var calls = parser.parse(
                "prefix"
                + "<start_function_call>call:first{payload:{enabled:true,label:<escape>a,b<escape>}}<end_function_call>"
                + "middle"
                + "<start_function_call>call:second{values:[1,2,3],ratio:1.25}<end_function_call>");

        assertEquals(List.of("first", "second"), calls.stream().map(
                FunctionGemmaToolCallParser.FunctionCall::name).toList());
        Map<?, ?> payload = (Map<?, ?>) calls.getFirst().arguments().get("payload");
        assertEquals(Boolean.TRUE, payload.get("enabled"));
        assertEquals("a,b", payload.get("label"));
        assertEquals(List.of(1L, 2L, 3L), calls.get(1).arguments().get("values"));
        assertEquals(new BigDecimal("1.25"), calls.get(1).arguments().get("ratio"));
    }

    @Test
    void preservesUnescapedIsoTimestampAsString() {
        var call = parser.parseCallBody(
                "call:create_event{datetime:2026-11-21T15:00:00,title:<escape>Planning<escape>}");
        assertEquals("2026-11-21T15:00:00", call.arguments().get("datetime"));
    }

    @Test
    void rejectsMalformedAndDuplicateArguments() {
        assertThrows(IllegalArgumentException.class,
                () -> parser.parse("<start_function_call>call:broken{x:1"));
        assertThrows(IllegalArgumentException.class,
                () -> parser.parseCallBody("call:duplicate{x:1,x:2}"));
        assertThrows(IllegalArgumentException.class,
                () -> parser.parseCallBody("call:bad name{x:1}"));
    }

    @Test
    void returnsEmptyListWhenOutputContainsNoFunctionCalls() {
        assertTrue(parser.parse("I can answer that directly.").isEmpty());
    }
}
