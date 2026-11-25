#!/usr/bin/env bash
set -euo pipefail

# Downloads the public google/embeddinggemma-2 vision checkpoint and runs
# Aljabr's opt-in compatibility test without Python, Transformers, or libtorch.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${ALJABR_EMBEDDING_GEMMA2_DIR:-"${REPO_ROOT}/.cache/models/embeddinggemma-2"}"
CHECKPOINT="${MODEL_DIR}/model.safetensors"
# Pin model weights and both configuration files to one immutable Hugging Face snapshot.
MODEL_REVISION="914f7f89142e33e77833254d9c9b90c3cef7303b"
MODEL_URL="https://huggingface.co/google/embeddinggemma-2/resolve/${MODEL_REVISION}"
CHECKPOINT_SHA256="197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"

usage() {
    echo "Usage: $0 [--forward]"
    echo "  --forward  Also execute the full 16-layer vision forward pass."
    echo
    echo "Override download location with ALJABR_EMBEDDING_GEMMA2_DIR."
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "Error: sha256sum or shasum is required to verify the official checkpoint." >&2
        return 1
    fi
}

verify_checkpoint() {
    local actual_sha
    actual_sha="$(sha256_file "$CHECKPOINT")"
    if [[ "$actual_sha" != "$CHECKPOINT_SHA256" ]]; then
        echo "Checkpoint SHA-256 mismatch: expected ${CHECKPOINT_SHA256}, got ${actual_sha}" >&2
        return 1
    fi
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
    local refresh="${2:-false}"
    local destination="${MODEL_DIR}/${name}"
    local temporary="${destination}.part"

    if [[ "$refresh" != true && -s "$destination" ]]; then
        if [[ "$name" != "model.safetensors" ]] || verify_checkpoint; then
            echo "Using verified cached ${name}"
            return
        fi
        echo "Cached ${name} failed integrity verification; downloading a fresh copy." >&2
        rm -f "$destination"
    fi

    echo "Downloading ${name}..."
    rm -f "$temporary"
    if ! curl --fail --location --retry 3 --retry-delay 2 \
        "${MODEL_URL}/${name}?download=true" \
        --output "$temporary"; then
        rm -f "$temporary"
        echo "Download failed: ${name}" >&2
        return 1
    fi
    if [[ ! -s "$temporary" ]]; then
        rm -f "$temporary"
        echo "Downloaded file is empty: ${name}" >&2
        return 1
    fi
    mv -f "$temporary" "$destination"
}

# These small files define the checkpoint architecture and preprocessing.
# Refresh them from the pinned snapshot rather than trusting an unverifiable cache.
download config.json true
download processor_config.json true
download model.safetensors
verify_checkpoint

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
