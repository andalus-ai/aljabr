package tech.kayys.aljabr.core.tool;

/**
 * Model-independent contract for extracting tool calls from model output.
 *
 * <p>Implementations belong to model or protocol adapters. The input is
 * untrusted model output, and the returned calls are descriptive data only:
 * implementations must not resolve, authorize, dispatch, or execute tools.</p>
 *
 * <p>Implementations should preserve source order and return recoverable
 * diagnostics rather than discard valid calls when the underlying protocol
 * permits recovery.</p>
 */
@FunctionalInterface
public interface ToolCallParser {

    /**
     * Parses model output into a structured result.
     *
     * @param modelOutput raw, untrusted model output; never {@code null}
     * @return extracted calls and any parsing diagnostics
     * @throws NullPointerException if {@code modelOutput} is {@code null}
     */
    ToolCallParseResult parseResult(String modelOutput);
}
