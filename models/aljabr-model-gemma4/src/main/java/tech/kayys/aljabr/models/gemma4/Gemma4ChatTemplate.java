/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.models.gemma4;

import java.util.List;
import java.util.Locale;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.tokenizer.ModelChatTemplate;

/**
 * Chat template formatter for Google Gemma 4 architecture.
 *
 * <p>Formats turns using the canonical Gemma 4 turn tags ({@code <|turn>}, {@code <turn|>})
 * and initializes an empty thought channel ({@code <|channel>thought\n<channel|>}) when
 * thinking mode is disabled.</p>
 */
public class Gemma4ChatTemplate implements ModelChatTemplate {

    @Override
    public boolean supports(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        String norm = modelType.trim().toLowerCase(Locale.ROOT).replace("-", "").replace("_", "");
        return norm.startsWith("gemma4");
    }

    @Override
    public String format(List<Message> messages) {
        var sb = new StringBuilder("<bos>");
        String start = "<|turn>";
        String end = "<turn|>";
        String firstUserPrefix = "";
        int startIdx = 0;
        if (!messages.isEmpty() && messages.get(0).getRole() == Message.Role.SYSTEM) {
            firstUserPrefix = messages.get(0).getContent().trim();
            if (!firstUserPrefix.isEmpty()) {
                firstUserPrefix += "\n\n";
            }
            startIdx = 1;
        }

        boolean firstRendered = true;
        for (int i = startIdx; i < messages.size(); i++) {
            var m = messages.get(i);
            String roleName = m.getRole() == Message.Role.ASSISTANT
                    ? "model"
                    : m.getRole().name().toLowerCase(Locale.ROOT);
            sb.append(start).append(roleName).append("\n");
            if (firstRendered && !firstUserPrefix.isEmpty() && m.getRole() == Message.Role.USER) {
                sb.append(firstUserPrefix);
            }
            sb.append(m.getContent().trim()).append(end).append("\n");
            firstRendered = false;
        }
        sb.append(start).append("model\n<|channel>thought\n<channel|>");
        return sb.toString();
    }
}
