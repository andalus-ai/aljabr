package tech.kayys.aljabr.models.functiongemma;

import tech.kayys.aljabr.core.tool.ToolDeclaration;
import tech.kayys.aljabr.core.tool.ToolSet;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.tokenizer.ModelChatTemplate;

import java.util.List;
import java.util.Objects;

/**
 * FunctionGemma conversation formatter.
 *
 * <p>Tool declarations belong in the developer turn. Function responses use
 * their dedicated control-token block and are not model chat turns.</p>
 */
public final class FunctionGemmaChatTemplate implements ModelChatTemplate {

    public static final String START_TURN = "<start_of_turn>";
    public static final String END_TURN = "<end_of_turn>";
    public static final String START_FUNCTION_DECLARATION = FunctionGemmaTokenizer.START_FUNCTION_DECLARATION;
    public static final String END_FUNCTION_DECLARATION = FunctionGemmaTokenizer.END_FUNCTION_DECLARATION;
    public static final String START_FUNCTION_RESPONSE = FunctionGemmaTokenizer.START_FUNCTION_RESPONSE;
    public static final String END_FUNCTION_RESPONSE = FunctionGemmaTokenizer.END_FUNCTION_RESPONSE;

    @Override
    public boolean supports(String modelType) {
        return modelType != null && modelType.equalsIgnoreCase("functiongemma");
    }

    /**
     * Builds the developer-turn body expected by FunctionGemma. Each
     * declaration is the schema payload (for example
     * {@code get_weather{location:{type:<escape>STRING<escape>}}}); string
     * values inside that payload must already use the {@code <escape>} token.
     */
    public String formatDeveloperContent(String instructions, List<String> declarations) {
        StringBuilder content = new StringBuilder(Objects.requireNonNullElse(instructions, ""));
        if (declarations != null) {
            for (String declaration : declarations) {
                if (declaration == null || declaration.isBlank()) {
                    throw new IllegalArgumentException("Function declaration must not be blank");
                }
                if (!content.isEmpty()) {
                    content.append('\n');
                }
                content.append(START_FUNCTION_DECLARATION)
                        .append(declaration)
                        .append(END_FUNCTION_DECLARATION);
            }
        }
        return content.toString();
    }

    /** Formats typed declarations directly into the developer-turn body. */
    public String formatDeveloperContent(String instructions, ToolDeclaration... tools) {
        return formatDeveloperContent(instructions, ToolSet.of(Objects.requireNonNull(tools, "tools")));
    }

    /** Formats an immutable, model-independent tool set using FunctionGemma's protocol. */
    public String formatDeveloperContent(String instructions, ToolSet tools) {
        Objects.requireNonNull(tools, "tools");
        FunctionGemmaToolDeclarationRenderer renderer = new FunctionGemmaToolDeclarationRenderer();
        List<String> declarations = new java.util.ArrayList<>(tools.size());
        for (ToolDeclaration tool : tools.declarations()) {
            declarations.add(renderer.render(tool));
        }
        return formatDeveloperContent(instructions, declarations);
    }

    @Override
    public String format(List<Message> messages) {
        Objects.requireNonNull(messages, "messages");
        StringBuilder prompt = new StringBuilder();
        for (Message message : messages) {
            if (message == null) {
                continue;
            }
            String content = Objects.toString(message.getContent(), "");
            switch (message.getRole()) {
                case SYSTEM -> appendTurn(prompt, "developer", content);
                case USER -> appendTurn(prompt, "user", content);
                case ASSISTANT -> appendTurn(prompt, "model", content);
                case FUNCTION, TOOL -> prompt.append(START_FUNCTION_RESPONSE)
                        .append(content)
                        .append(END_FUNCTION_RESPONSE)
                        .append('\n');
            }
        }
        prompt.append(START_TURN).append("model\n");
        return prompt.toString();
    }

    private static void appendTurn(StringBuilder prompt, String role, String content) {
        prompt.append(START_TURN).append(role).append('\n')
                .append(content)
                .append(END_TURN).append('\n');
    }
}
