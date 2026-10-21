#!/bin/bash

# Aljabr Deployment Utility
# Facilitates multi-module project versioning, git tagging, and artifact deployment.

set -e

# Default values
NEW_VERSION=""
REPO_URL=""
SNAPSHOT_REPO_URL=""
DRY_RUN=false
SKIP_TESTS=true
GIT_TAG=false
GIT_PUSH=true      # Enabled by default
RUN_GRADLE=true
SKIP_COMMIT=false
FORCE=false
LOCAL_MAVEN=false
PUBLISH_GITHUB=false
CREATE_RELEASE=false
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-}"

# Keep local SSH aliases (e.g. github-bhangun) for convenience, but use the canonical GitHub
# repository URL for remote operations and GitHub API-aware publishing.
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

resolve_github_repository() {
    local remote_url
    remote_url="$(git config --get remote.origin.url || true)"
    if [[ -z "$remote_url" ]]; then
        echo "${GITHUB_REPOSITORY:-}"
        return 0
    fi

    remote_url="$(canonicalize_github_remote "$remote_url")"
    if [[ "$remote_url" =~ ^git@github\.com:([^/]+)/([^/]+?)(\.git)?$ ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    elif [[ "$remote_url" =~ ^https?://github\.com/([^/]+)/([^/]+?)(\.git)?$ ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    else
        echo "${GITHUB_REPOSITORY:-}"
    fi
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root (script is in scripts/ or root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo -e "${BLUE}--------------------------------------------------${NC}"
    echo -e "${GREEN} Aljabr Deployment Utility ${NC}"
    echo -e "${BLUE}--------------------------------------------------${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -v, --version <new-version>      Update project version (e.g., 0.1.0 or 1.0.1-SNAPSHOT)"
    echo "  -r, --repo <url>                 Override release repository URL"
    echo "  -s, --snapshot-repo <url>        Override snapshot repository URL"
    echo "  -t, --tag                        Create git tag (auto-enabled for release versions)"
    echo "  --no-push                        Skip pushing commits and tags to remote"
    echo "  -f, --force                      Force actions (e.g., overwrite existing tags)"
    echo "  --no-build                       Skip Gradle build and deployment"
    echo "  --keep-tests                     Do not skip tests during build"
    echo "  --skip-commit                    Skip git commit after version update"
    echo "  --local-maven, --local           Publish only to Maven Local (skips remote publish)"
    echo "  --github                         Publish only to GitHub Packages"
    echo "  --release, --create-release      Create a GitHub Release (via gh CLI or API) for the tag"
    echo "  --dry-run                        Show commands without executing them"
    echo "  -h, --help                       Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -v 0.1.0                      # Version, build/deploy, tag, and PUSH"
    echo "  $0 -v 0.1.0 --release            # Version, build/deploy, tag, PUSH, and create GitHub Release"
    echo "  $0 -v 0.1.0 --no-push            # Build/deploy and tag locally only"
    echo "  $0 -v 1.0.0-SNAPSHOT             # Version and build/deploy (no tag, but PUSHES commit)"
    echo "  $0 --local                       # Publish to Maven Local only"
    echo "  $0 --github                      # Publish to GitHub Packages only"
    echo ""
    exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--version) NEW_VERSION="$2"; shift ;;
        -r|--repo) REPO_URL="$2"; shift ;;
        -s|--snapshot-repo) SNAPSHOT_REPO_URL="$2"; shift ;;
        -t|--tag) GIT_TAG=true ;;
        -p|--push) GIT_PUSH=true ;; # Keep for compat
        --no-push) GIT_PUSH=false ;;
        -f|--force) FORCE=true ;;
        --no-build|--no-jar) RUN_GRADLE=false ;;
        --keep-tests) SKIP_TESTS=false ;;
        --skip-commit) SKIP_COMMIT=true ;;
        --local-maven|--local) LOCAL_MAVEN=true ;;
        --github) PUBLISH_GITHUB=true ;;
        --release|--create-release) CREATE_RELEASE=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; usage ;;
    esac
    shift
done

# Navigate to project root
cd "$PROJECT_ROOT"

REMOTE_ORIGIN_CANONICAL="$(canonicalize_github_remote "$(git config --get remote.origin.url || true)")"
if [[ -n "$REMOTE_ORIGIN_CANONICAL" ]]; then
    GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-$(resolve_github_repository)}"
fi
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-bhangun/aljabr}"
export GITHUB_REPOSITORY

echo -e "${BLUE}--------------------------------------------------${NC}"
echo -e "${GREEN} Aljabr Deployment Utility ${NC}"
echo -e "${BLUE}--------------------------------------------------${NC}"
echo ""

# Helper function for dry-run execution
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        eval "$@"
    fi
}

# Version Management
if [ -n "$NEW_VERSION" ]; then
    echo -e "${BLUE}>>> Updating version to: ${GREEN}$NEW_VERSION${NC}"
    
    # Check if version is a release (not SNAPSHOT)
    if [[ "$NEW_VERSION" != *"-SNAPSHOT" ]]; then
        echo -e "${YELLOW}ℹ Release version detected (not SNAPSHOT)${NC}"
        GIT_TAG=true
    fi
    
    # Update version in build.gradle.kts
    echo -e "${BLUE}>>> Updating Gradle configurations...${NC}"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} sed -i.bak -e \"s/extra\\[\\\"aljabrVersion\\\"\\] = \\\".*\\\"/extra\\[\\\"aljabrVersion\\\"\\] = \\\"$NEW_VERSION\\\"/\" build.gradle.kts"
    else
        sed -i.bak -e "s/extra\[\"aljabrVersion\"\] = \".*\"/extra\[\"aljabrVersion\"\] = \"$NEW_VERSION\"/" build.gradle.kts
        rm -f build.gradle.kts.bak
    fi
    
    # Commit version changes
    COMMIT_SUCCESS=false
    if [ "$SKIP_COMMIT" = false ]; then
        echo -e "${BLUE}>>> Committing version changes...${NC}"
        run_cmd "git add build.gradle.kts"
        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY-RUN]${NC} git commit -m \"chore: bump version to $NEW_VERSION\""
        elif git commit -m "chore: bump version to $NEW_VERSION"; then
            COMMIT_SUCCESS=true
        else
            echo -e "${YELLOW}⚠ No changes to commit or not a git repo${NC}"
        fi
    fi
    
    # Create git tag if requested
    if [ "$GIT_TAG" = true ]; then
        TAG_NAME="$NEW_VERSION"
        echo -e "${BLUE}>>> Creating git tag: ${GREEN}$TAG_NAME${NC}"
        
        # Check if tag already exists
        if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
            if [ "$FORCE" = true ]; then
                echo -e "${YELLOW}⚠ Tag $TAG_NAME already exists. Forcing recreation...${NC}"
                run_cmd "git tag -d \"$TAG_NAME\""
                run_cmd "git tag -a \"$TAG_NAME\" -m \"Release $NEW_VERSION\""
                echo -e "${GREEN}✓ Git tag force-recreated: $TAG_NAME${NC}"
            else
                echo -e "${YELLOW}⚠ Tag $TAG_NAME already exists. Skipping tag creation (use -f to force).${NC}"
            fi
        else
            run_cmd "git tag -a \"$TAG_NAME\" -m \"Release $NEW_VERSION\""
            echo -e "${GREEN}✓ Git tag created: $TAG_NAME${NC}"
        fi
    fi

    # Push to remote if enabled
    if [ "$GIT_PUSH" = true ]; then
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
        REMOTE_PUSH_URL="$(git config --get remote.origin.url || true)"
        if [[ -z "$REMOTE_PUSH_URL" ]]; then
            REMOTE_PUSH_URL="origin"
        fi
        SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa_bhangun}"
        export GIT_SSH_COMMAND="ssh -i \"$SSH_KEY_PATH\" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
        echo -e "${BLUE}>>> Pushing to remote (branch: $CURRENT_BRANCH, url: $REMOTE_PUSH_URL)...${NC}"

        if [[ "$REMOTE_PUSH_URL" == git@* || "$REMOTE_PUSH_URL" == https://* ]]; then
            run_cmd "git push \"$REMOTE_PUSH_URL\" $CURRENT_BRANCH"
        else
            run_cmd "git push origin $CURRENT_BRANCH"
        fi

        # Push the tag if it was created
        if [ "$GIT_TAG" = true ]; then
            echo -e "${BLUE}>>> Pushing git tag to remote...${NC}"
            if [[ "$REMOTE_PUSH_URL" == git@* || "$REMOTE_PUSH_URL" == https://* ]]; then
                if [ "$FORCE" = true ]; then
                    run_cmd "git push \"$REMOTE_PUSH_URL\" :refs/tags/\"$TAG_NAME\" || true"
                    run_cmd "git push \"$REMOTE_PUSH_URL\" \"$TAG_NAME\""
                else
                    run_cmd "git push \"$REMOTE_PUSH_URL\" \"$TAG_NAME\""
                fi
            else
                if [ "$FORCE" = true ]; then
                    run_cmd "git push origin :refs/tags/\"$TAG_NAME\" || true"
                    run_cmd "git push origin \"$TAG_NAME\""
                else
                    run_cmd "git push origin \"$TAG_NAME\""
                fi
            fi
        fi
        echo -e "${GREEN}✓ Pushed to remote successfully${NC}"

        # Create GitHub Release if requested
        if [ "$CREATE_RELEASE" = true ] && [ "$GIT_TAG" = true ]; then
            echo -e "${BLUE}>>> Creating GitHub Release for ${GREEN}$TAG_NAME${NC}...${NC}"
            RELEASE_TITLE="Release $NEW_VERSION"
            IS_PRERELEASE=false
            if [[ "$NEW_VERSION" == *"-SNAPSHOT"* || "$NEW_VERSION" == *"-alpha"* || "$NEW_VERSION" == *"-beta"* || "$NEW_VERSION" == *"-rc"* ]]; then
                IS_PRERELEASE=true
            fi

            GH_CMD="gh release create \"$TAG_NAME\" --repo \"$GITHUB_REPOSITORY\" --title \"$RELEASE_TITLE\" --generate-notes"
            if [ "$IS_PRERELEASE" = true ]; then
                GH_CMD="$GH_CMD --prerelease"
            fi

            if command -v gh >/dev/null 2>&1; then
                if [ "$DRY_RUN" = true ]; then
                    echo -e "${YELLOW}[DRY-RUN]${NC} $GH_CMD"
                else
                    eval "$GH_CMD" || echo -e "${YELLOW}⚠ gh release create failed or release already exists${NC}"
                fi
            elif [ -n "${GITHUB_TOKEN:-}" ]; then
                echo -e "${BLUE}>>> Creating GitHub Release via GitHub API...${NC}"
                API_PAYLOAD="{\"tag_name\":\"$TAG_NAME\",\"name\":\"$RELEASE_TITLE\",\"prerelease\":$IS_PRERELEASE,\"generate_release_notes\":true}"
                if [ "$DRY_RUN" = true ]; then
                    echo -e "${YELLOW}[DRY-RUN]${NC} curl -X POST https://api.github.com/repos/$GITHUB_REPOSITORY/releases -d '$API_PAYLOAD'"
                else
                    curl -s -f -X POST \
                        -H "Authorization: token $GITHUB_TOKEN" \
                        -H "Accept: application/vnd.github.v3+json" \
                        "https://api.github.com/repos/$GITHUB_REPOSITORY/releases" \
                        -d "$API_PAYLOAD" >/dev/null || echo -e "${YELLOW}⚠ GitHub Release API creation failed or release already exists${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ Neither 'gh' CLI nor GITHUB_TOKEN is available. Skipped GitHub Release creation.${NC}"
            fi
        fi
    fi
    
    echo -e "${GREEN}✓ Version updated to $NEW_VERSION${NC}"
    echo ""
fi

# Build and Deployment
if [ "$RUN_GRADLE" = true ]; then
    if [ "$PUBLISH_GITHUB" = true ]; then
        if [ -z "${GITHUB_ACTOR:-}" ] || [ -z "${GITHUB_TOKEN:-}" ]; then
            echo -e "${RED}Error: GITHUB_ACTOR and GITHUB_TOKEN must be set to publish to GitHub Packages.${NC}"
            exit 1
        fi
        GRADLE_TASKS="publishAllPublicationsToGitHubPackagesRepository"
    elif [ "$LOCAL_MAVEN" = true ]; then
        GRADLE_TASKS="publishToMavenLocal"
    else
        # Default: publish to local Maven, and to GitHub Packages if credentials are set
        if [ -n "${GITHUB_ACTOR:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
            GRADLE_TASKS="publishToMavenLocal publishAllPublicationsToGitHubPackagesRepository"
        else
            echo -e "${YELLOW}ℹ GITHUB_ACTOR / GITHUB_TOKEN not set; publishing to Maven Local only.${NC}"
            GRADLE_TASKS="publishToMavenLocal"
        fi
    fi

    GRADLE_ARGS="--no-daemon $GRADLE_TASKS"

    if [ "$SKIP_TESTS" = true ]; then
        GRADLE_ARGS="$GRADLE_ARGS -x test"
    fi

    if [ -n "$REPO_URL" ]; then
        GRADLE_ARGS="$GRADLE_ARGS -Pdeployment.repo.url=$REPO_URL"
    fi

    if [ -n "$SNAPSHOT_REPO_URL" ]; then
        GRADLE_ARGS="$GRADLE_ARGS -Pdeployment.snapshot.repo.url=$SNAPSHOT_REPO_URL"
    fi

    echo -e "${BLUE}>>> Running deployment: ./gradlew $GRADLE_ARGS${NC}"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} ./gradlew $GRADLE_ARGS"
    else
        ./gradlew $GRADLE_ARGS
    fi
else
    echo -e "${YELLOW}⚠ Gradle build and deployment skipped (--no-build)${NC}"
fi

echo ""
echo -e "${BLUE}--------------------------------------------------${NC}"
echo -e "${GREEN} Deployment process completed!${NC}"
echo -e "${BLUE}--------------------------------------------------${NC}"

# Summary
if [ -n "$NEW_VERSION" ]; then
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo -e "  Version:  ${GREEN}$NEW_VERSION${NC}"
    if [ "$GIT_TAG" = true ]; then
        echo -e "  Git Tag:  ${GREEN}$NEW_VERSION${NC}"
    fi
    echo -e "  Pushed:   $( [ "$GIT_PUSH" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${YELLOW}No${NC}" )"
    if [ "$CREATE_RELEASE" = true ]; then
        echo -e "  Release:  ${GREEN}Created ($NEW_VERSION)${NC}"
    fi
    if [ "$RUN_GRADLE" = true ]; then
        echo -e "  Build:    ${GREEN}$GRADLE_TASKS${NC}"
    else
        echo -e "  Build:    ${YELLOW}Skipped${NC}"
    fi
    echo ""
fi
