#!/usr/bin/env bash
# =============================================================================
# check-boundaries.sh — Aljabr Architectural Boundary Enforcement
# =============================================================================
# Runs four static-analysis checks against the Aljabr source tree to
# enforce module-level architectural boundaries.  Each check emits ✅ / ❌.
# All hard failures are collected; the script exits non-zero only at the very
# end so every check is always reported.
#
# Check 4 is a WARNING-only check — it prints offending filenames but does
# NOT contribute to a non-zero exit code.
#
# Usage:
#   ./scripts/check-boundaries.sh          # from anywhere
#   bash scripts/check-boundaries.sh       # explicit interpreter
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the Aljabr root regardless of where the caller invoked the script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALJABR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ALJABR_ROOT}"

# ---------------------------------------------------------------------------
# Color helpers (tput when available, plain fallback)
# ---------------------------------------------------------------------------
if command -v tput &>/dev/null && tput setaf 1 &>/dev/null; then
  RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"
  YELLOW="$(tput setaf 3)"
  BOLD="$(tput bold)"
  RESET="$(tput sgr0)"
else
  RED="" GREEN="" YELLOW="" BOLD="" RESET=""
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS_ICON="✅"
FAIL_ICON="❌"
WARN_ICON="⚠️ "

pass()  { echo "${GREEN}${PASS_ICON}  $*${RESET}"; }
fail()  { echo "${RED}${FAIL_ICON}  $*${RESET}"; }
warn()  { echo "${YELLOW}${WARN_ICON} $*${RESET}"; }
header(){ echo "${BOLD}$*${RESET}"; }

# Tracks how many checks pass/fail
TOTAL=0
PASSED=0
FAILURES=()

# Hard-failure check — exits non-zero on violation
run_check() {
  local description="$1"
  shift
  TOTAL=$(( TOTAL + 1 ))

  local output
  output=$("$@" 2>/dev/null || true)

  if [[ -z "${output}" ]]; then
    pass "${description}"
    PASSED=$(( PASSED + 1 ))
  else
    fail "${description}"
    while IFS= read -r line; do
      echo "    ${RED}${line}${RESET}"
    done <<< "${output}"
    FAILURES+=("${description}")
  fi
}

# Warning-only check — never adds to FAILURES
run_warn_check() {
  local description="$1"
  shift

  local output
  output=$("$@" 2>/dev/null || true)

  if [[ -z "${output}" ]]; then
    pass "${description}"
  else
    warn "${description}"
    while IFS= read -r line; do
      echo "    ${YELLOW}${line}${RESET}"
    done <<< "${output}"
  fi
}

# =============================================================================
# Banner
# =============================================================================
echo
echo "${BOLD}=============================================================${RESET}"
echo "${BOLD}  Aljabr — Architectural Boundary Checks${RESET}"
echo "${BOLD}  Root: ${ALJABR_ROOT}${RESET}"
echo "${BOLD}=============================================================${RESET}"
echo

# =============================================================================
# Check 1 — No backend module depends on another non-CPU backend module
# =============================================================================
# Backend modules are isolated by hardware; they may only depend on core or
# the universal CPU fallback (:backend:cpu:*), never across competing accelerators
# (e.g. CUDA must not depend on ROCm or Metal).
header "Check 1: Non-CPU cross-backend dependency isolation"
run_check \
  "Cross-hardware backend dependency isolation" \
  bash -c 'for kts in $(find backend -name "build.gradle.kts"); do
    backend_family=$(echo "$kts" | cut -d/ -f2)
    grep -rn "project(\":backend:" "$kts" 2>/dev/null | grep -v ":backend:${backend_family}:" | grep -v ":backend:cpu:" || true
  done'

# =============================================================================
# Check 2 — No core module declares a backend package
# =============================================================================
# Core layers must not contain backend-namespaced types; that would invert the
# dependency graph and expose backend concerns to all consumers of core.
header "Check 2: No core module declares a backend package"
run_check \
  "Backend package leakage into core source" \
  grep -rn '^package tech\.kayys\.aljabr\.backend' --include='*.java' core/

# =============================================================================
# Check 3 — No stale / non-existent module references
# =============================================================================
# These module names were either renamed or never shipped; their presence in
# build scripts indicates unresolved copy-paste or incomplete migrations.
header "Check 3: No stale non-existent module references"
run_check \
  "Stale module references (aljabr-spi-provider / aljabr-model-runner / aljabr-engine dependency)" \
  bash -c 'grep -rn "aljabr-spi-provider\|aljabr-model-runner\|name = \"aljabr-engine\"" --include="*.kts" . | grep -v "rootProject.name" || true'

# =============================================================================
# Check 4 (WARN ONLY) — All published modules have explicit group + version
# =============================================================================
# Missing group/version in a published module's build script causes accidental
# publication under wrong coordinates.  This is a warning, not a hard failure,
# because it may legitimately apply to modules that inherit from a root script.
header "Check 4 (warn only): All published modules declare explicit group + version"
run_warn_check \
  "Modules missing 'group = \"tech.kayys.aljabr\"'" \
  bash -c "grep -rL 'group = \"tech.kayys.aljabr\"' \
    core/*/build.gradle.kts backend/*/*/build.gradle.kts 2>/dev/null \
    | grep -v '.gradle-sandbox'"

# =============================================================================
# Summary
# =============================================================================
echo
echo "${BOLD}=============================================================${RESET}"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "${GREEN}${BOLD}  ${PASSED}/${TOTAL} checks passed — all boundaries clean.${RESET}"
  echo "${BOLD}=============================================================${RESET}"
  echo
  exit 0
else
  echo "${RED}${BOLD}  ${PASSED}/${TOTAL} checks passed — ${#FAILURES[@]} boundary violation(s) found:${RESET}"
  for f in "${FAILURES[@]}"; do
    echo "${RED}    • ${f}${RESET}"
  done
  echo "${BOLD}=============================================================${RESET}"
  echo
  exit 1
fi
