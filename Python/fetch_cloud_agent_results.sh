#!/usr/bin/env bash
# Clone branch and extract the FULL cloud-agent MAB results bundle.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ruanhq-StatsML/Causal_Objective_Permutation_Test.git}"
BRANCH="${BRANCH:-cursor/eps-greedy-robustness-d30-9eb2}"
DIR="${DIR:-Causal_Objective_Permutation_Test_cloud_results}"

if [[ -d "$DIR/.git" ]]; then
  cd "$DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH" || git reset --hard "origin/$BRANCH"
else
  git clone -b "$BRANCH" --single-branch "$REPO_URL" "$DIR"
  cd "$DIR"
fi

BUNDLE="Python/cloud_agent_mab_results_bundle.tar.gz"
test -f "$BUNDLE"

# Extract into workspace-like layout
mkdir -p cloud_agent_workspace/Python
tar -xzf "$BUNDLE" -C cloud_agent_workspace/Python
echo "Extracted $(tar -tzf "$BUNDLE" | wc -l) files -> $(pwd)/cloud_agent_workspace/Python/"
ls cloud_agent_workspace/Python/mab_multi_shift_d30_by_config.csv
ls cloud_agent_workspace/Python/mab_multi_shift_d30_shift_all_methods_bands.png
echo "Agent: https://cursor.com/agents/bc-2fb3ca4b-d04a-46b8-b039-656735ce9eb2"
