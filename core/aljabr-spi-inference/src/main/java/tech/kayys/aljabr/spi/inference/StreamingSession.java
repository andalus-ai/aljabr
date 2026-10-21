package tech.kayys.aljabr.spi.inference;

import io.smallrye.mutiny.Multi;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.auth.ApiKeyConstants;
import tech.kayys.aljabr.spi.context.RequestContext;

public record StreamingSession(
        String sessionId,
        String modelId,
        RequestContext requestContext,
        Multi<Message> stream) {
    public String apiKey() {
        if (requestContext.apiKey() == null || requestContext.apiKey().isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return requestContext.apiKey();
    }
}
