#!/usr/bin/env bash
set -euo pipefail

execute=0
resume=0
authorize_budget_policy_amendment=0
expected_new_protocol_hash=""
while (( $# > 0 )); do
  case "$1" in
    --execute) execute=1 ;;
    --resume) resume=1 ;;
    --authorize-budget-policy-amendment) authorize_budget_policy_amendment=1 ;;
    --expected-new-protocol-hash)
      shift
      if (( $# == 0 )); then
        echo "--expected-new-protocol-hash requires a value" >&2
        exit 2
      fi
      expected_new_protocol_hash="$1"
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
if (( resume == 1 && execute == 0 )); then
  echo "--resume is only valid together with --execute" >&2
  exit 2
fi
if (( authorize_budget_policy_amendment == 1 )); then
  if (( execute == 0 || resume == 0 )) || [[ -z "$expected_new_protocol_hash" ]]; then
    echo "budget-policy amendment requires --execute, --resume, and --expected-new-protocol-hash" >&2
    exit 2
  fi
elif [[ -n "$expected_new_protocol_hash" ]]; then
  echo "--expected-new-protocol-hash requires --authorize-budget-policy-amendment" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SKILLGEN_PYTHON:-/home/linyuanjing/.venvs/skillgen-benchmarking/bin/python}"
data_root="$repo_root/data/skillsbench-families/family-13-economic-financial-quant/holdout-reserves-at-risk-calc"
config="$repo_root/config.skillsbench.deepseek-v4-flash-family-pilot.yaml"
run_root="${SKILLGEN_PILOT_RUN_ROOT:-$repo_root/artifacts/skillsbench/family-pilot-deepseek-v4-flash}"
if [[ "$run_root" != /* ]]; then
  run_root="$repo_root/$run_root"
fi

export SKILLSBENCH_ROOT="${SKILLSBENCH_ROOT:-/home/linyuanjing/skillsbench-v1.1}"
export SKILLSBENCH_JOBS_ROOT="${SKILLSBENCH_JOBS_ROOT:-/home/linyuanjing/skillgen-jobs/deepseek-family13-heldout-pilot}"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export SKILLGEN_CHAT_PROVIDER="deepseek"
export LITELLM_LOCAL_MODEL_COST_MAP="True"

if [[ ! -x "$python_bin" ]]; then
  echo "SkillGen Python environment is missing: $python_bin" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop/WSL integration is not ready; start Docker Desktop first." >&2
  exit 1
fi

common=(
  --family-id family-13-economic-financial-quant
  --induction-dataset "$data_root/induction.json"
  --verification-dataset "$data_root/verification.json"
  --heldout-dataset "$data_root/heldout.json"
  --manifest "$data_root/protocol_manifest.json"
  --config "$config"
  --run-root "$run_root"
  --budget-cny 120
)

require_keys=()
if (( execute == 1 )); then
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set DEEPSEEK_API_KEY and OPENAI_API_KEY in this WSL shell first." >&2
    exit 1
  fi
  require_keys=(--require-keys)

  # The pilot is intentionally admitted only in the announced off-peak
  # windows, even before the future tariff takes effect.
  if [[ "${SKILLGEN_ALLOW_PEAK_LAUNCH:-0}" != "1" ]]; then
    clock="$(TZ=Asia/Shanghai date +%H%M)"
    clock=$((10#$clock))
    if (( (clock >= 900 && clock < 1200) || (clock >= 1400 && clock < 1800) )); then
      echo "DeepSeek peak window is active; wait until 12:00-14:00 or after 18:00." >&2
      exit 1
    fi
  fi
fi

"$python_bin" "$repo_root/scripts/check_skillsbench_family_environment.py" \
  "${common[@]}" "${require_keys[@]}"

# Always print the final frozen no-API plan immediately before the paid gate.
"$python_bin" "$repo_root/scripts/run_skillsbench_family.py" "${common[@]}"

if (( execute == 0 )); then
  echo "Preflight and dry-run passed. Re-run with --execute after reviewing the plan."
  exit 0
fi

paid=(--execute)
if (( resume == 1 )); then
  paid+=(--resume)
fi
if (( authorize_budget_policy_amendment == 1 )); then
  paid+=(
    --authorize-budget-policy-amendment
    --expected-new-protocol-hash "$expected_new_protocol_hash"
  )
fi
"$python_bin" "$repo_root/scripts/run_skillsbench_family.py" \
  "${common[@]}" "${paid[@]}"
