#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Publishing Aljabr to GitHub Packages..."

canonicalize_github_remote() {
  local remote_url="${1:-}"
  if [[ -z "$remote_url" ]]; then
    return 0
  fi
  remote_url="${remote_url//github-bhangun/github.com}"
  remote_url="${remote_url//github-ngoding/github.com}"
  remote_url="${remote_url//github-*/github.com}"
  echo "$remote_url"
}

if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
  if git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    raw_remote="$(git -C "$PROJECT_ROOT" config --get remote.origin.url || true)"
    remote_url="$(canonicalize_github_remote "$raw_remote")"
    if [[ "$remote_url" =~ ^git@github\.com:([^/]+)/([^/]+?)(\.git)?$ ]]; then
      GITHUB_REPOSITORY="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    elif [[ "$remote_url" =~ ^https?://github\.com/([^/]+)/([^/]+?)(\.git)?$ ]]; then
      GITHUB_REPOSITORY="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    fi
  fi
fi

GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-bhangun/aljabr}"
export GITHUB_REPOSITORY

GITHUB_REPOSITORY_OWNER="${GITHUB_REPOSITORY%%/*}"
export GITHUB_REPOSITORY_OWNER

if [[ -z "${GITHUB_ACTOR:-}" || -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Error: GITHUB_ACTOR and GITHUB_TOKEN environment variables must be set."
  echo "Use a PAT with 'read:packages' and 'write:packages' scopes."
  echo "Example: export GITHUB_ACTOR=bhangun; export GITHUB_TOKEN=..."
  exit 1
fi

cd "$PROJECT_ROOT"

echo "Repository: ${GITHUB_REPOSITORY}"
echo "Publishing to: https://maven.pkg.github.com/${GITHUB_REPOSITORY}"

# GitHub Packages Maven endpoints require https://maven.pkg.github.com/OWNER/REPOSITORY
./gradlew --no-daemon publishAllPublicationsToGitHubPackagesRepository

echo "✅ Successfully published to GitHub Packages."
