package tech.kayys.aljabr.spi;

import java.util.Objects;

/**
 * Minimal, dependency-free chat message for tokenizer/template formatting in
 * the Aljabr foundation.
 *
 * <p>
 * This is intentionally a lightweight value type with no serving-layer
 * dependencies
 * (no Mutiny, no Jakarta validation, no tool-call wiring). It lives in the
 * math/tokenizer
 * layer so that {@code ChatTemplateFormatter} can format prompts without
 * pulling in any
 * serving-engine concepts.
 *
 * <p>
 * The serving engine (Aljabr) defines its own richer {@code Message} type that
 * adds
 * multi-modal content parts, tool calls, etc. and bridges to this type where
 * needed.
 * @author bhangun
 */
public final class Message {

    public enum Role {
        SYSTEM,
        USER,
        ASSISTANT,
        FUNCTION,
        TOOL
    }

    private final Role role;
    private final String content;

    public Message(Role role, String content) {
        this.role = Objects.requireNonNull(role, "role");
        this.content = content;
    }

    public Role getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }

    // ── Factory methods ──────────────────────────────────────────────────────

    public static Message system(String content) {
        return new Message(Role.SYSTEM, content);
    }

    public static Message user(String content) {
        return new Message(Role.USER, content);
    }

    public static Message assistant(String content) {
        return new Message(Role.ASSISTANT, content);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Message m))
            return false;
        return role == m.role && Objects.equals(content, m.content);
    }

    @Override
    public int hashCode() {
        return Objects.hash(role, content);
    }

    @Override
    public String toString() {
        String preview = content != null && content.length() > 50
                ? content.substring(0, 50) + "..."
                : content;
        return "Message{role=" + role + ", content='" + preview + "'}";
    }
}
