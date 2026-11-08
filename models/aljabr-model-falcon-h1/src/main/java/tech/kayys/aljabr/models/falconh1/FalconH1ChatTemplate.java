/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.models.falconh1;

import java.util.List;
import java.util.Locale;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.tokenizer.ModelChatTemplate;

/**
 * Chat template formatter for Falcon-H1 architecture (ChatML format).
 */
public class FalconH1ChatTemplate implements ModelChatTemplate {

    @Override
    public boolean supports(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        String norm = modelType.trim().toLowerCase(Locale.ROOT).replace("-", "").replace("_", "");
        return norm.startsWith("falconh1") || norm.startsWith("falcon");
    }

    @Override
    public String format(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "<|im_start|>assistant\n";
        }
        var sb = new StringBuilder();
        for (var m : messages) {
            String roleName = m.getRole().name().toLowerCase(Locale.ROOT);
            sb.append("<|im_start|>").append(roleName).append("\n");
            sb.append(m.getContent()).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }
}
