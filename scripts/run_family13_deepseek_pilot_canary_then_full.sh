#!/usr/bin/env bash
set -euo pipefail

resume=0
authorize_budget_policy_amendment=0
expected_new_protocol_hash=""
while (( $# > 0 )); do
  case "$1" in
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
if (( authorize_budget_policy_amendment == 1 )); then
  if (( resume == 0 )) || [[ -z "$expected_new_protocol_hash" ]]; then
    echo "budget-policy amendment requires --resume and --expected-new-protocol-hash" >&2
    exit 2
  fi
elif [[ -n "$expected_new_protocol_hash" ]]; then
  echo "--expected-new-protocol-hash requires --authorize-budget-policy-amendment" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SKILLGEN_PYTHON:-/home/linyuanjing/.venvs/skillgen-benchmarking/bin/python}"
data_root="$repo_root/data/skillsbench-families/family-13-economic-financial-quant/holdout-reserves-at-risk-calc"
run_root="${SKILLGEN_PILOT_RUN_ROOT:-$repo_root/artifacts/skillsbench/family-pilot-deepseek-v4-flash-v6}"
if [[ "$run_root" != /* ]]; then
  run_root="$repo_root/$run_root"
fi

launcher_dir="$run_root/background-launcher"
mkdir -p "$launcher_dir"
chmod 700 "$launcher_dir"
exec >>"$launcher_dir/run.log" 2>&1

started_at="$(date --iso-8601=seconds)"
printf '%s\n' "$$" >"$launcher_dir/pid"
printf '%s\n' "started_at=$started_at" "stage=starting" >"$launcher_dir/status"

finish() {
  rc=$?
  finished_at="$(date --iso-8601=seconds)"
  printf '%s\n' \
    "started_at=$started_at" \
    "finished_at=$finished_at" \
    "stage=finished" \
    "exit_code=$rc" >"$launcher_dir/status"
}
trap finish EXIT

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Required provider keys are missing from the transient process environment." >&2
  exit 1
fi
if [[ ! -x "$python_bin" ]]; then
  echo "SkillGen Python environment is missing: $python_bin" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop/WSL integration is not ready." >&2
  exit 1
fi

if [[ "${SKILLGEN_ALLOW_PEAK_LAUNCH:-0}" != "1" ]]; then
  clock="$(TZ=Asia/Shanghai date +%H%M)"
  clock=$((10#$clock))
  if (( (clock >= 900 && clock < 1200) || (clock >= 1400 && clock < 1800) )); then
    echo "DeepSeek peak window is active; refusing to launch." >&2
    exit 1
  fi
fi

export SKILLGEN_PILOT_RUN_ROOT="$run_root"
export SKILLSBENCH_ROOT="${SKILLSBENCH_ROOT:-/home/linyuanjing/skillsbench-v1.1}"
export SKILLSBENCH_JOBS_ROOT="${SKILLSBENCH_JOBS_ROOT:-/home/linyuanjing/skillgen-jobs/deepseek-family13-heldout-pilot}"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export SKILLGEN_CHAT_PROVIDER="deepseek"
export LITELLM_LOCAL_MODEL_COST_MAP="True"

if (( authorize_budget_policy_amendment == 1 )); then
  printf '%s\n' \
    "started_at=$started_at" \
    "stage=canary-skipped-authorized-budget-policy-amendment" \
    >"$launcher_dir/status"
  echo "Skipping the old-protocol canary receipt for the explicitly authorized budget-only amendment."
else
  printf '%s\n' "started_at=$started_at" "stage=canary" >"$launcher_dir/status"
  "$python_bin" "$repo_root/scripts/run_skillsbench_family_canary.py" \
    --induction-dataset "$data_root/induction.json" \
    --manifest "$data_root/protocol_manifest.json" \
    --config "$repo_root/config.skillsbench.deepseek-v4-flash-family-pilot.yaml" \
    --instance-id "family-13-economic-financial-quant::shock-analysis-demand::induction::r000" \
    --run-root "$run_root" \
    --budget-cny 120 \
    --execute
fi

printf '%s\n' "started_at=$started_at" "stage=formal-family-pilot" >"$launcher_dir/status"
formal_args=(--execute)
if (( resume == 1 )); then
  formal_args+=(--resume)
fi
if (( authorize_budget_policy_amendment == 1 )); then
  formal_args+=(
    --authorize-budget-policy-amendment
    --expected-new-protocol-hash "$expected_new_protocol_hash"
  )
fi
bash "$repo_root/scripts/run_family13_deepseek_pilot.sh" "${formal_args[@]}"
