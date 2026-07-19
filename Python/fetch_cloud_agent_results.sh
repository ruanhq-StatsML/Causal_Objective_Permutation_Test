#!/usr/bin/env bash
# Fetch the full cloud-agent MAB results bundle from the feature branch.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ruanhq-StatsML/Causal_Objective_Permutation_Test.git}"
BRANCH="${BRANCH:-cursor/eps-greedy-robustness-d30-9eb2}"
DIR="${DIR:-Causal_Objective_Permutation_Test_cloud_results}"

if [[ -d "$DIR/.git" ]]; then
  cd "$DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
else
  git clone -b "$BRANCH" --single-branch "$REPO_URL" "$DIR"
  cd "$DIR"
fi

BUNDLE="Python/cloud_agent_mab_results_bundle.tar.gz"
if [[ ! -f "$BUNDLE" ]]; then
  echo "Bundle not found: $BUNDLE" >&2
  exit 1
fi

mkdir -p cloud_agent_results
tar -xzf "$BUNDLE" -C cloud_agent_results --strip-components=0
echo "Extracted to: $(pwd)/cloud_agent_results/Python/"
ls cloud_agent_results/Python/mab_multi_shift_d30_by_config.csv
ls cloud_agent_results/Python/mab_results_multi/ | head
echo "Done. Agent page: https://cursor.com/agents/bc-2fb3ca4b-d04a-46b8-b039-656735ce9eb2"
