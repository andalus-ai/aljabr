#!/usr/bin/env bash
set -euo pipefail

# Downloads the public google/embeddinggemma-2 vision checkpoint and runs
# Aljabr's opt-in compatibility test without Python, Transformers, or libtorch.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${ALJABR_EMBEDDING_GEMMA2_DIR:-"${REPO_ROOT}/.cache/models/embeddinggemma-2"}"
CHECKPOINT="${MODEL_DIR}/model.safetensors"
MODEL_URL="https://huggingface.co/google/embeddinggemma-2/resolve/main"
CHECKPOINT_SHA256="197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"

usage() {
    echo "Usage: $0 [--forward]"
    echo "  --forward  Also execute the full 16-layer vision forward pass."
    echo
    echo "Override download location with ALJABR_EMBEDDING_GEMMA2_DIR."
}

RUN_FORWARD=false
while (($#)); do
    case "$1" in
        --forward) RUN_FORWARD=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
mkdir -p "$MODEL_DIR"

download() {
    local name="$1"
    local destination="${MODEL_DIR}/${name}"
    if [[ -s "$destination" ]]; then
        echo "Using cached ${name}"
        return
    fi
    echo "Downloading ${name}..."
    curl --fail --location --retry 3 --retry-delay 2 \
        "${MODEL_URL}/${name}?download=true" \
        --output "${destination}.part"
    mv "${destination}.part" "$destination"
}

download config.json
download processor_config.json
download model.safetensors

if command -v sha256sum >/dev/null 2>&1; then
    actual_sha="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
    if [[ "$actual_sha" != "$CHECKPOINT_SHA256" ]]; then
        echo "Checkpoint SHA-256 mismatch: expected ${CHECKPOINT_SHA256}, got ${actual_sha}" >&2
        exit 1
    fi
elif command -v shasum >/dev/null 2>&1; then
    actual_sha="$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')"
    if [[ "$actual_sha" != "$CHECKPOINT_SHA256" ]]; then
        echo "Checkpoint SHA-256 mismatch: expected ${CHECKPOINT_SHA256}, got ${actual_sha}" >&2
        exit 1
    fi
else
    echo "Warning: neither sha256sum nor shasum is available; skipping checkpoint digest verification." >&2
fi

cd "$REPO_ROOT"
if [[ "$RUN_FORWARD" == true ]]; then
    echo "Running checkpoint mapping and full vision forward test..."
    ALJABR_EMBEDDING_GEMMA2_CHECKPOINT="$CHECKPOINT" \
        ALJABR_EMBEDDING_GEMMA2_RUN_FORWARD=true \
        ./gradlew :engine:safetensor:aljabr-safetensor-embedding-gemma2:test \
        --tests 'tech.kayys.aljabr.safetensor.loader.EmbeddingGemma2OfficialCheckpointCompatibilityTest'
else
    echo "Running checkpoint mapping compatibility test..."
    ALJABR_EMBEDDING_GEMMA2_CHECKPOINT="$CHECKPOINT" \
        ./gradlew :engine:safetensor:aljabr-safetensor-embedding-gemma2:test \
        --tests 'tech.kayys.aljabr.safetensor.loader.EmbeddingGemma2OfficialCheckpointCompatibilityTest.mapsPublishedCheckpointAndBuildsImagePipeline'
fi
