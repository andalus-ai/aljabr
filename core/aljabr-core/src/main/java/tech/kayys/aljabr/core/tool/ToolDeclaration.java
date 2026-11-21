package tech.kayys.aljabr.core.tool;

import java.util.Map;

/**
 * Model-independent description of a callable tool.
 *
 * <p>This contract contains metadata and a JSON-Schema-compatible parameter
 * schema only. It does not register, authorize, invoke, or otherwise execute
 * the tool. Model-specific modules adapt declarations to their prompt or
 * structured-output protocols.</p>
 */
public interface ToolDeclaration {

    /** Stable tool identifier used by the caller/runtime to resolve a call. */
    String name();

    /** Human-readable purpose presented to a model. */
    String description();

    /**
     * Immutable JSON-Schema-compatible object schema describing tool arguments.
     * The root schema must describe an object.
     */
    Map<String, Object> parameterSchema();
}
