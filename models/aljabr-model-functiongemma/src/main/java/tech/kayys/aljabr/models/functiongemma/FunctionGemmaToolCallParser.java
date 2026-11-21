package tech.kayys.aljabr.models.functiongemma;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Parses the native FunctionGemma function-call stream.
 *
 * <p>Calls have the form
 * {@code <start_function_call>call:name{key:<escape>string<escape>}<end_function_call>}.
 * Escape markers delimit string values, so commas and braces inside strings
 * do not interfere with the outer argument grammar.</p>
 *
 * <p>This parser only extracts model output. It never invokes a tool or
 * treats parsed arguments as trusted input.</p>
 */
public final class FunctionGemmaToolCallParser {
    private static final String START = FunctionGemmaTokenizer.START_FUNCTION_CALL;
    private static final String END = FunctionGemmaTokenizer.END_FUNCTION_CALL;
    private static final String ESCAPE = FunctionGemmaTokenizer.ESCAPE;

    /** Parses every complete function call in a model output, preserving output order. */
    public List<FunctionCall> parse(String output) {
        Objects.requireNonNull(output, "output");
        List<FunctionCall> calls = new ArrayList<>();
        int cursor = 0;
        while (true) {
            int start = output.indexOf(START, cursor);
            if (start < 0) break;
            int bodyStart = start + START.length();
            int end = output.indexOf(END, bodyStart);
            if (end < 0) throw malformed("unterminated function call", start);
            calls.add(parseCallBody(output.substring(bodyStart, end), start));
            cursor = end + END.length();
        }
        return List.copyOf(calls);
    }

    /** Parses one call body without surrounding control tokens. */
    public FunctionCall parseCallBody(String body) {
        Objects.requireNonNull(body, "body");
        return parseCallBody(body, 0);
    }

    private FunctionCall parseCallBody(String body, int offset) {
        String value = body.trim();
        if (!value.startsWith("call:")) throw malformed("function call must begin with 'call:'", offset);
        int open = value.indexOf('{', 5);
        if (open < 0 || !value.endsWith("}")) {
            throw malformed("function call must contain a braced argument object", offset);
        }
        String name = value.substring(5, open).trim();
        if (name.isEmpty() || !isIdentifier(name)) throw malformed("invalid function name", offset + 5);
        String argumentBody = value.substring(open + 1, value.length() - 1);
        return new FunctionCall(name, parseArguments(argumentBody, offset + open + 1));
    }

    private Map<String, Object> parseArguments(String source, int offset) {
        Map<String, Object> arguments = new LinkedHashMap<>();
        if (source.isBlank()) return Collections.unmodifiableMap(arguments);
        for (String entry : splitTopLevel(source, ',', offset)) {
            if (entry.isBlank()) continue;
            int separator = findTopLevel(entry, ':');
            if (separator < 1) throw malformed("argument must have key:value form", offset);
            String key = entry.substring(0, separator).trim();
            if (!isIdentifier(key)) throw malformed("invalid argument name '" + key + "'", offset);
            if (arguments.containsKey(key)) throw malformed("duplicate argument '" + key + "'", offset);
            String rawValue = entry.substring(separator + 1).trim();
            arguments.put(key, parseValue(rawValue, offset + separator + 1));
        }
        return Collections.unmodifiableMap(arguments);
    }

    private Object parseValue(String raw, int offset) {
        if (raw.startsWith(ESCAPE)) {
            if (!raw.endsWith(ESCAPE) || raw.length() < ESCAPE.length() * 2) {
                throw malformed("unterminated escaped string value", offset);
            }
            return raw.substring(ESCAPE.length(), raw.length() - ESCAPE.length());
        }
        if (raw.equals("None") || raw.equals("null")) return null;
        if (raw.equalsIgnoreCase("true")) return Boolean.TRUE;
        if (raw.equalsIgnoreCase("false")) return Boolean.FALSE;
        if (raw.startsWith("{") && raw.endsWith("}")) {
            return parseArguments(raw.substring(1, raw.length() - 1), offset + 1);
        }
        if (raw.startsWith("[") && raw.endsWith("]")) {
            String inner = raw.substring(1, raw.length() - 1);
            if (inner.isBlank()) return List.of();
            List<Object> values = new ArrayList<>();
            for (String item : splitTopLevel(inner, ',', offset + 1)) {
                values.add(parseValue(item.trim(), offset + 1));
            }
            return Collections.unmodifiableList(values);
        }
        if (raw.matches("[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?")) {
            try {
                BigDecimal number = new BigDecimal(raw);
                try {
                    return number.longValueExact();
                } catch (ArithmeticException ignored) {
                    return number;
                }
            } catch (NumberFormatException ignored) {
                // Preserve unknown protocol atoms as strings.
            }
        }
        if (raw.isEmpty()) throw malformed("empty argument value", offset);
        return raw;
    }

    private static List<String> splitTopLevel(String source, char delimiter, int offset) {
        List<String> parts = new ArrayList<>();
        int start = 0, braces = 0, brackets = 0;
        boolean escaped = false;
        for (int index = 0; index < source.length(); index++) {
            if (source.startsWith(ESCAPE, index)) {
                escaped = !escaped;
                index += ESCAPE.length() - 1;
                continue;
            }
            if (escaped) continue;
            char current = source.charAt(index);
            if (current == '{') braces++;
            else if (current == '}') braces--;
            else if (current == '[') brackets++;
            else if (current == ']') brackets--;
            else if (current == delimiter && braces == 0 && brackets == 0) {
                parts.add(source.substring(start, index));
                start = index + 1;
            }
            if (braces < 0 || brackets < 0) throw malformed("unbalanced argument delimiters", offset + index);
        }
        if (escaped) throw malformed("unterminated escaped string value", offset + source.length());
        if (braces != 0 || brackets != 0) throw malformed("unbalanced argument delimiters", offset + source.length());
        parts.add(source.substring(start));
        return parts;
    }

    private static int findTopLevel(String source, char delimiter) {
        int braces = 0, brackets = 0;
        boolean escaped = false;
        for (int index = 0; index < source.length(); index++) {
            if (source.startsWith(ESCAPE, index)) {
                escaped = !escaped;
                index += ESCAPE.length() - 1;
                continue;
            }
            if (escaped) continue;
            char current = source.charAt(index);
            if (current == '{') braces++;
            else if (current == '}') braces--;
            else if (current == '[') brackets++;
            else if (current == ']') brackets--;
            else if (current == delimiter && braces == 0 && brackets == 0) return index;
        }
        return -1;
    }

    private static boolean isIdentifier(String value) {
        return value.matches("[A-Za-z_][A-Za-z0-9_.-]*");
    }

    private static IllegalArgumentException malformed(String message, int offset) {
        return new IllegalArgumentException("Malformed FunctionGemma call at offset " + offset + ": " + message);
    }

    /** Immutable, untrusted function call extracted from model output. */
    public record FunctionCall(String name, Map<String, Object> arguments) {
        public FunctionCall {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(arguments, "arguments");
            arguments = Collections.unmodifiableMap(new LinkedHashMap<>(arguments));
        }
    }
}
