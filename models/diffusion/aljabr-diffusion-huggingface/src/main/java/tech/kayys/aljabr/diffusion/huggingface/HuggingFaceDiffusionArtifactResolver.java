package tech.kayys.aljabr.diffusion.huggingface;

import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionContext;
import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionException;
import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolver;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.ResolvedDiffusionArtifact;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Materializes one file from a Hugging Face model repository.
 *
 * <p>Locations use {@code hf://owner/repository/path/to/file?revision=main}.
 * Mutable revisions are resolved once to a commit SHA; downloads and cache
 * paths are then pinned to that immutable commit. Authentication is read from
 * {@code HF_TOKEN} and is never included in diagnostics or cache keys.</p>
 */
public final class HuggingFaceDiffusionArtifactResolver implements DiffusionArtifactResolver {
    private static final Pattern COMMIT_SHA = Pattern.compile("\\"sha\\"\\s*:\\s*\\"([0-9a-fA-F]{40})\\"");
    private static final Pattern SHA256 = Pattern.compile("[0-9a-fA-F]{64}");

    private final HttpClient client;
    private final String endpoint;
    private final String token;

    public HuggingFaceDiffusionArtifactResolver() {
        this(HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build(),
                "https://huggingface.co", System.getenv("HF_TOKEN"));
    }

    public HuggingFaceDiffusionArtifactResolver(HttpClient client, String endpoint, String token) {
        this.client = Objects.requireNonNull(client, "client");
        this.endpoint = trimTrailingSlash(Objects.requireNonNull(endpoint, "endpoint"));
        this.token = token == null || token.isBlank() ? null : token;
    }

    @Override public String scheme() { return "hf"; }

    @Override
    public boolean supports(DiffusionComponentArtifact artifact) {
        if (artifact == null) return false;
        try { return "hf".equalsIgnoreCase(URI.create(artifact.location()).getScheme())
                && parse(artifact.location()) != null; }
        catch (RuntimeException ignored) { return false; }
    }

    @Override
    public ResolvedDiffusionArtifact resolve(
            DiffusionComponentArtifact artifact, DiffusionArtifactResolutionContext context) {
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(context, "context");
        final Location location;
        try { location = parse(artifact.location()); }
        catch (RuntimeException e) { throw failure(artifact, "Invalid Hugging Face artifact location", e); }

        try {
            String commit = resolveCommit(location, context.requestTimeout());
            Path target = cachePath(context.cacheDirectory(), location, commit);
            if (Files.isRegularFile(target)) {
                String actual = sha256(target);
                verifyDeclaredChecksum(artifact, actual);
                return new ResolvedDiffusionArtifact(artifact, target, commit, actual, true);
            }

            Files.createDirectories(target.getParent());
            Path temporary = Files.createTempFile(target.getParent(), ".download-", ".part");
            try {
                HttpResponse<Path> response = sendFile(downloadRequest(location, commit, context.requestTimeout()), temporary);
                if (response.statusCode() / 100 != 2) {
                    throw new IOException("Hugging Face file request returned HTTP " + response.statusCode());
                }
                String actual = sha256(temporary);
                verifyDeclaredChecksum(artifact, actual);
                String lfsHash = response.headers().firstValue("X-Linked-Etag")
                        .or(() -> response.headers().firstValue("ETag"))
                        .map(HuggingFaceDiffusionArtifactResolver::normalizeEtag)
                        .filter(SHA256.asPredicate()).orElse(null);
                if (lfsHash != null && !actual.equalsIgnoreCase(lfsHash)) {
                    throw new IOException("Downloaded file does not match the repository LFS SHA-256");
                }
                try {
                    Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
                } catch (java.nio.file.AtomicMoveNotSupportedException ignored) {
                    Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
                }
                return new ResolvedDiffusionArtifact(artifact, target, commit, actual, false);
            } finally {
                Files.deleteIfExists(temporary);
            }
        } catch (DiffusionArtifactResolutionException e) {
            throw e;
        } catch (Exception e) {
            throw failure(artifact, "Unable to resolve Hugging Face artifact", e);
        }
    }

    private String resolveCommit(Location location, Duration timeout) throws IOException, InterruptedException {
        if (location.revision().matches("[0-9a-fA-F]{40}")) return location.revision().toLowerCase(Locale.ROOT);
        String url = endpoint + "/api/models/" + encodePath(location.repository())
                + "/revision/" + encode(location.revision());
        HttpResponse<String> response = sendText(request(url, timeout).GET().build());
        if (response.statusCode() / 100 != 2) throw new IOException("Revision lookup returned HTTP " + response.statusCode());
        Matcher matcher = COMMIT_SHA.matcher(response.body());
        if (!matcher.find()) throw new IOException("Revision response did not contain an immutable commit SHA");
        return matcher.group(1).toLowerCase(Locale.ROOT);
    }

    private HttpRequest downloadRequest(Location location, String commit, Duration timeout) {
        String url = endpoint + "/" + encodePath(location.repository())
                + "/resolve/" + commit + "/" + encodePath(location.file());
        return request(url, timeout).GET().build();
    }

    private HttpRequest.Builder request(String url, Duration timeout) {
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url)).timeout(timeout);
        if (token != null) builder.header("Authorization", "Bearer " + token);
        return builder;
    }

    private HttpResponse<String> sendText(HttpRequest request) throws IOException, InterruptedException {
        return client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
    }

    private HttpResponse<Path> sendFile(HttpRequest request, Path destination) throws IOException, InterruptedException {
        return client.send(request, HttpResponse.BodyHandlers.ofFile(destination));
    }

    private static Path cachePath(Path root, Location location, String commit) {
        Path base = root.resolve("huggingface").resolve(location.owner()).resolve(location.repositoryName())
                .resolve(commit).normalize();
        Path target = base.resolve(location.file()).normalize();
        if (!target.startsWith(base)) throw new IllegalArgumentException("Artifact path escapes repository cache");
        return target;
    }

    private static Location parse(String value) {
        URI uri = URI.create(value);
        if (!"hf".equalsIgnoreCase(uri.getScheme()) || uri.getRawQuery() == null
                && uri.getRawPath() == null) throw new IllegalArgumentException("Expected hf://owner/repository/file");
        String owner = uri.getHost();
        if (owner == null || owner.isBlank()) throw new IllegalArgumentException("Repository owner is missing");
        String path = uri.getPath();
        if (path == null) throw new IllegalArgumentException("Repository file path is missing");
        String[] parts = path.replaceFirst("^/", "").split("/");
        if (parts.length < 2) throw new IllegalArgumentException("Expected repository name and file path");
        String repo = parts[0];
        StringBuilder file = new StringBuilder();
        for (int i = 1; i < parts.length; i++) {
            if (parts[i].isBlank() || parts[i].equals(".") || parts[i].equals(".."))
                throw new IllegalArgumentException("Unsafe repository file path");
            if (file.length() > 0) file.append('/');
            file.append(parts[i]);
        }
        String revision = "main";
        if (uri.getRawQuery() != null) {
            for (String parameter : uri.getRawQuery().split("&")) {
                String[] pair = parameter.split("=", 2);
                if (pair[0].equals("revision") && pair.length == 2 && !pair[1].isBlank())
                    revision = java.net.URLDecoder.decode(pair[1], StandardCharsets.UTF_8);
            }
        }
        if (revision.contains("/") || revision.equals(".") || revision.equals(".."))
            throw new IllegalArgumentException("Revision must be a single ref name or commit SHA");
        return new Location(owner, owner + "/" + repo, repo, file.toString(), revision);
    }

    private static String encodePath(String value) {
        return java.util.Arrays.stream(value.split("/")).map(HuggingFaceDiffusionArtifactResolver::encode)
                .collect(java.util.stream.Collectors.joining("/"));
    }
    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8).replace("+", "%20");
    }
    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(path)) {
                byte[] buffer = new byte[1024 * 1024];
                int read;
                while ((read = input.read(buffer)) >= 0) if (read > 0) digest.update(buffer, 0, read);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) { throw new IllegalStateException("SHA-256 is unavailable", e); }
    }
    private static void verifyDeclaredChecksum(DiffusionComponentArtifact artifact, String actual) throws IOException {
        if (artifact.sha256() != null && !artifact.sha256().equalsIgnoreCase(actual))
            throw new IOException("Downloaded artifact does not match its declared SHA-256");
    }
    private static String normalizeEtag(String value) {
        String result = value == null ? "" : value.trim();
        if (result.startsWith("W/")) result = result.substring(2);
        if (result.length() >= 2 && result.startsWith("\\"") && result.endsWith("\\""))
            result = result.substring(1, result.length() - 1);
        return result;
    }
    private static String trimTrailingSlash(String value) {
        return value.endsWith("/") ? value.substring(0, value.length() - 1) : value;
    }
    private static DiffusionArtifactResolutionException failure(
            DiffusionComponentArtifact artifact, String message, Throwable cause) {
        return new DiffusionArtifactResolutionException(artifact, message, cause);
    }
    private record Location(String owner, String repository, String repositoryName, String file, String revision) { }
}
