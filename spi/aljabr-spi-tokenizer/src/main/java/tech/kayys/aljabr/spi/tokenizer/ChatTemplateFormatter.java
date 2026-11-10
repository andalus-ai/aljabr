/*
 * Aljabr Foundational Infrastructure
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.tokenizer;

import java.util.List;
import java.util.Locale;
import java.util.ServiceLoader;
import java.util.concurrent.CopyOnWriteArrayList;
import tech.kayys.aljabr.spi.Message;

/**
 * Dispatches message formatting to discovered {@link ModelChatTemplate} providers,
 * falling back to ChatML if no provider matches.
 */
public final class ChatTemplateFormatter {

    private static final List<ModelChatTemplate> PROVIDERS = new CopyOnWriteArrayList<>();

    static {
        loadProviders();
    }

    private ChatTemplateFormatter() {}

    public static synchronized void reloadProviders() {
        PROVIDERS.clear();
        loadProviders();
    }

    private static void loadProviders() {
        for (ModelChatTemplate template : ServiceLoader.load(ModelChatTemplate.class)) {
            PROVIDERS.add(template);
        }
    }

    /**
     * Formats conversation messages into a prompt string using the registered
     * provider for {@code modelType}.
     */
    public static String format(List<Message> messages, String modelType) {
        if (modelType != null && !modelType.isBlank()) {
            for (ModelChatTemplate provider : PROVIDERS) {
                if (provider.supports(modelType)) {
                    return provider.format(messages);
                }
            }
        }
        return formatChatML(messages);
    }

    /**
     * Returns true if any registered provider supports the model type, or if it is ChatML.
     */
    public static boolean supportsModelType(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return false;
        }
        for (ModelChatTemplate provider : PROVIDERS) {
            if (provider.supports(modelType)) {
                return true;
            }
        }
        return "chatml".equalsIgnoreCase(modelType.trim());
    }

    /**
     * Generic ChatML fallback format.
     */
    public static String formatChatML(List<Message> messages) {
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
