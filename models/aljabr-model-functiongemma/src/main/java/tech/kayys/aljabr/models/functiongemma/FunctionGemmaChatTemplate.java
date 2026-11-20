package tech.kayys.aljabr.models.functiongemma;

import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.tokenizer.ModelChatTemplate;

import java.util.List;
import java.util.Objects;

/**
 * FunctionGemma conversation formatter.
 *
 * <p>Function declarations are supplied as developer-message content and
 * should contain the model's {@code <start_function_declaration>} and
 * {@code <end_function_declaration>} control tokens. Function-call payloads
 * are preserved verbatim so the tokenizer can resolve the dedicated tokens.</p>
 */
public final class FunctionGemmaChatTemplate implements ModelChatTemplate {

    public static final String START_TURN = "<start_of_turn>";
    public static final String END_TURN = "<end_of_turn>";
    public static final String START_FUNCTION_RESPONSE = "<start_function_response>";
    public static final String END_FUNCTION_RESPONSE = "<end_function_response>";

    @Override
    public boolean supports(String modelType) {
        return modelType != null && modelType.equalsIgnoreCase("functiongemma");
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
                case FUNCTION, TOOL -> appendTurn(prompt, "model",
                        START_FUNCTION_RESPONSE + content + END_FUNCTION_RESPONSE);
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
