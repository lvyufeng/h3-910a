#!/usr/bin/env bash
# Read-only preflight for the two four-card MiniMax H3 instances on ts-217.
#
# Usage:
#   bash scripts/check_ts217_env.sh [--json]
#
# Important environment:
#   H3_MODEL_ROOT          Model roots, colon-separated.
#                          Default: /home/mseco/models
#   H3_MODEL_MANIFEST      JSON list with destination, size and sha256 fields.
#   H3_HASH_MODELS         full|none (default: full)
#   H3_RUNTIME_ROOT        Directory checked for free space.
#   H3_CANN_HOME           CANN toolkit root. Default: auto-detect.
#   H3_EXPECTED_CANN       Expected CANN version. Default: 9.2.0
#   H3_EXPECTED_DEVICES    Expected visible NPU count. Default: 8
#   H3_REQUIRE_TOPO        "1" if topology inspection is mandatory. Default: 0
#   H3_REQUIRE_ATC         "1" if atc is mandatory. Default: 1
#
# The script never sources CANN setup scripts and never creates files. Hash
# verification reads the model files and can take several minutes.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/check_ts217_env.sh [--json]

Read-only checks: OS, driver, npu-smi, CANN, atc, torch_npu, topology, HBM,
CPU, RAM, disk space, model sizes and optionally model SHA256 values.

Configuration is supplied through H3_* environment variables. See the header
of this script and artifacts/design/harness_report.md for the full interface.
EOF
}

case "${1:-text}" in
    -h|--help)
        usage
        exit 0
        ;;
    text)
        SCAN_MODE=text
        ;;
    --json|json)
        SCAN_MODE=json
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

readonly SCAN_MODE
readonly EXPECTED_CANN="${H3_EXPECTED_CANN:-9.2.0}"
readonly EXPECTED_DEVICES="${H3_EXPECTED_DEVICES:-8}"
readonly HASH_MODE="${H3_HASH_MODELS:-full}"
readonly REQUIRE_TOPO="${H3_REQUIRE_TOPO:-0}"
readonly REQUIRE_ATC="${H3_REQUIRE_ATC:-1}"
readonly MODEL_ROOT="${H3_MODEL_ROOT:-/home/mseco/models}"
readonly MANIFEST="${H3_MODEL_MANIFEST:-}"
readonly RUNTIME_ROOT="${H3_RUNTIME_ROOT:-${H3_INSTANCE_ROOT:-/mnt/workspace/h3-ascend/ts217}}"
readonly CHECK_TIMEOUT="${H3_CHECK_TIMEOUT:-30}"
readonly MIN_CPU_CORES="${H3_MIN_CPU_CORES:-0}"
readonly MIN_RAM_GIB="${H3_MIN_RAM_GIB:-0}"
readonly MIN_DISK_GIB="${H3_MIN_DISK_GIB:-0}"
readonly MIN_RUNTIME_DISK_GIB="${H3_MIN_RUNTIME_DISK_GIB:-${MIN_DISK_GIB}}"

if [[ ! "$EXPECTED_DEVICES" =~ ^[1-9][0-9]*$ ]]; then
    echo "H3_EXPECTED_DEVICES must be a positive integer" >&2
    exit 2
fi
if [[ ! "$HASH_MODE" =~ ^(full|none)$ ]]; then
    echo "H3_HASH_MODELS must be full or none" >&2
    exit 2
fi
if [[ ! "$REQUIRE_TOPO" =~ ^[01]$ ]]; then
    echo "H3_REQUIRE_TOPO must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$REQUIRE_ATC" =~ ^[01]$ ]]; then
    echo "H3_REQUIRE_ATC must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$CHECK_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "H3_CHECK_TIMEOUT must be a positive integer in seconds" >&2
    exit 2
fi
for pair in \
    "H3_MIN_CPU_CORES=${MIN_CPU_CORES}" \
    "H3_MIN_RAM_GIB=${MIN_RAM_GIB}" \
    "H3_MIN_DISK_GIB=${MIN_DISK_GIB}" \
    "H3_MIN_RUNTIME_DISK_GIB=${MIN_RUNTIME_DISK_GIB}"; do
    value="${pair#*=}"
    if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "${pair%%=*} must be a non-negative number" >&2
        exit 2
    fi
done

# Keep Python imports from creating __pycache__ files in site-packages.
export PYTHONDONTWRITEBYTECODE=1

declare -A STATUS=()
declare -A DETAIL=()
FAILURES=0
WARNINGS=0

record() {
    local key="$1"
    local status="$2"
    local detail="${3:-}"
    STATUS["$key"]="$status"
    DETAIL["$key"]="$detail"
    case "$status" in
        FAIL) FAILURES=$((FAILURES + 1)) ;;
        WARN) WARNINGS=$((WARNINGS + 1)) ;;
    esac
}

first_line() {
    local value="${1//$'\r'/}"
    printf '%s\n' "${value%%$'\n'*}"
}

limit_lines() {
    local limit="$1"
    awk -v limit="$limit" '
        NR <= limit { print }
        NR == limit + 1 { print "... output truncated ..." }
    '
}

json_string() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\b'/\\b}"
    value="${value//$'\f'/\\f}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '"%s"' "$value"
}

run_probe() {
    local -a command_line=("$@")
    if command -v timeout >/dev/null 2>&1; then
        LC_ALL=C timeout "$CHECK_TIMEOUT" "${command_line[@]}" 2>&1
    else
        LC_ALL=C "${command_line[@]}" 2>&1
    fi
}

os_name="unknown"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    os_name="$(set +u; . /etc/os-release; printf '%s %s' "${NAME:-}" "${VERSION_ID:-}")"
fi
kernel="$(uname -r 2>/dev/null || printf 'unknown')"
arch="$(uname -m 2>/dev/null || printf 'unknown')"
record os_info OK "${os_name}; kernel=${kernel}; arch=${arch}"

npu_smi_text=""
if command -v npu-smi >/dev/null 2>&1; then
    npu_smi_text="$(run_probe npu-smi info || true)"
    if [[ -n "$npu_smi_text" ]]; then
        record npu_smi OK "$(printf '%s\n' "$npu_smi_text" | limit_lines 40)"
    else
        record npu_smi FAIL "npu-smi produced no output"
    fi
else
    record npu_smi FAIL "npu-smi not found in PATH"
fi

driver_version=""
declare -a driver_candidates=(
    /usr/local/Ascend/driver/version.info
    /usr/local/Ascend/driver/version.cfg
)
shopt -s nullglob
for candidate in \
    /usr/local/Ascend/*/driver/version.info \
    /usr/local/Ascend/*/driver/version.cfg; do
    driver_candidates+=("$candidate")
done
for candidate in "${driver_candidates[@]}"; do
    if [[ -r "$candidate" ]]; then
        driver_version="$(
            grep -Eim1 '(^|[[:space:]])(Version|version)[[:space:]]*=' \
                "$candidate" || true
        )"
        driver_version="${driver_version#*=}"
        driver_version="${driver_version//[[:space:]]/}"
        [[ -n "$driver_version" ]] && break
    fi
done
if [[ -z "$driver_version" && -n "$npu_smi_text" ]]; then
    driver_version="$(
        printf '%s\n' "$npu_smi_text" |
            grep -Eo 'Version:[[:space:]]*[0-9][^[:space:]]*' |
            head -n1 |
            sed 's/Version:[[:space:]]*//' || true
    )"
fi
if [[ -n "$driver_version" ]]; then
    record driver OK "$driver_version"
else
    record driver FAIL "driver version not found"
fi

cann_home="${H3_CANN_HOME:-${ASCEND_HOME_PATH:-}}"
if [[ -z "$cann_home" ]]; then
    for candidate in \
        "/usr/local/Ascend/ascend-toolkit/${EXPECTED_CANN}" \
        /usr/local/Ascend/ascend-toolkit/latest \
        "$HOME/Ascend/ascend-toolkit/${EXPECTED_CANN}" \
        "$HOME/Ascend/ascend-toolkit/latest"; do
        if [[ -d "$candidate" ]]; then
            cann_home="$candidate"
            break
        fi
    done
fi

cann_version=""
if [[ -n "$cann_home" && -d "$cann_home" ]]; then
    for candidate in \
        "$cann_home/version.cfg" \
        "$cann_home/compiler/version.info" \
        "$cann_home/version.info" \
        "$cann_home/../version.cfg" \
        "$cann_home/../version.info"; do
        if [[ -r "$candidate" ]]; then
            cann_version="$(
                grep -Eim1 '^(Version|version|version_dir)[[:space:]]*=' \
                    "$candidate" || true
            )"
            cann_version="${cann_version#*=}"
            cann_version="${cann_version//[[:space:]]/}"
            [[ -n "$cann_version" ]] && break
        fi
    done
    if [[ -z "$cann_version" ]]; then
        marker="$(
            find "$cann_home" -maxdepth 3 -type f \
                \( -name 'version.cfg' -o -name 'version.info' \) \
                -print -quit 2>/dev/null || true
        )"
        if [[ -n "$marker" && -r "$marker" ]]; then
            cann_version="$(
                grep -Eim1 '(Version|version|version_dir)[[:space:]]*=' \
                    "$marker" || true
            )"
            cann_version="${cann_version#*=}"
            cann_version="${cann_version//[[:space:]]/}"
        fi
    fi
    if [[ "$cann_version" == "$EXPECTED_CANN" ]]; then
        record cann OK "${cann_home}; version=${cann_version}"
    elif [[ -n "$cann_version" ]]; then
        record cann WARN \
            "${cann_home}; found=${cann_version}; expected=${EXPECTED_CANN}"
    else
        record cann FAIL "${cann_home}; version marker not found"
    fi
else
    record cann FAIL "CANN toolkit not found"
fi

if [[ -n "$cann_home" && -r "$cann_home/set_env.sh" ]]; then
    record cann_set_env OK "$cann_home/set_env.sh"
else
    record cann_set_env FAIL "set_env.sh not found under CANN root"
fi

atc_bin=""
if command -v atc >/dev/null 2>&1; then
    atc_bin="$(command -v atc)"
elif [[ -n "$cann_home" ]]; then
    for candidate in "$cann_home/bin/atc" "$cann_home/compiler/bin/atc"; do
        if [[ -x "$candidate" ]]; then
            atc_bin="$candidate"
            break
        fi
    done
fi
if [[ -n "$atc_bin" ]]; then
    atc_output="$(run_probe "$atc_bin" --version || true)"
    if [[ -n "$atc_output" ]]; then
        record atc OK "${atc_bin}; $(first_line "$atc_output")"
    elif [[ "$REQUIRE_ATC" == "1" ]]; then
        record atc FAIL "${atc_bin}; atc returned no version"
    else
        record atc WARN "${atc_bin}; atc returned no version"
    fi
elif [[ "$REQUIRE_ATC" == "1" ]]; then
    record atc FAIL "atc not found"
else
    record atc WARN "atc not found"
fi

declare -a device_paths=()
for path in /dev/davinci[0-9]*; do
    [[ -e "$path" ]] && device_paths+=("$path")
done
device_count="${#device_paths[@]}"
if [[ "$device_count" -eq "$EXPECTED_DEVICES" ]]; then
    record devices OK \
        "${device_count} device nodes: ${device_paths[*]:-none}"
else
    record devices FAIL \
        "found=${device_count}; expected=${EXPECTED_DEVICES}; ${device_paths[*]:-none}"
fi

python_bin="${H3_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    record runtime FAIL "python not found: $python_bin"
    record torch FAIL "cannot inspect without Python runtime"
    record torch_npu FAIL "cannot inspect without Python runtime"
    record torch_npu_devices FAIL "cannot inspect without Python runtime"
    record model_files FAIL "cannot inspect without Python runtime"
else
    python_probe="$(
        "$python_bin" - "$EXPECTED_DEVICES" <<'PY' 2>&1 || true
import platform
import sys

result = {
    "python": platform.python_version(),
    "torch": "",
    "torch_npu": "",
    "npu_available": "",
    "device_count": "",
    "device_names": "",
    "error": "",
}
try:
    import torch

    result["torch"] = str(torch.__version__)
    try:
        import torch_npu

        result["torch_npu"] = str(
            getattr(torch_npu, "__version__", "unknown")
        )
        result["npu_available"] = str(bool(torch.npu.is_available())).lower()
        if torch.npu.is_available():
            count = int(torch.npu.device_count())
            result["device_count"] = str(count)
            result["device_names"] = "|".join(
                str(torch.npu.get_device_name(index))
                for index in range(count)
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

for key, value in result.items():
    print(f"{key}={str(value).replace(chr(10), ' ').replace(chr(13), ' ')}")
print(f"expected={sys.argv[1]}")
PY
    )"
    python_version=""
    torch_version=""
    torch_npu_version=""
    npu_available=""
    torch_device_count=""
    torch_device_names=""
    python_error=""
    while IFS= read -r line; do
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            python) python_version="$value" ;;
            torch) torch_version="$value" ;;
            torch_npu) torch_npu_version="$value" ;;
            npu_available) npu_available="$value" ;;
            device_count) torch_device_count="$value" ;;
            device_names) torch_device_names="$value" ;;
            error) python_error="$value" ;;
        esac
    done <<< "$python_probe"
    runtime_detail="python=${python_version:-unknown}"
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        runtime_detail+="; ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
    fi
    if [[ -n "$python_version" ]]; then
        record runtime OK "$runtime_detail"
    else
        record runtime FAIL \
            "${runtime_detail}; Python probe returned no version: ${python_error:-unknown error}"
    fi
    if [[ -n "$torch_version" ]]; then
        record torch OK "$torch_version"
    else
        record torch FAIL "${python_error:-torch import failed}"
    fi
    if [[ -n "$torch_npu_version" ]]; then
        record torch_npu OK "$torch_npu_version"
    else
        record torch_npu FAIL "${python_error:-torch_npu import failed}"
    fi
    if [[ "$npu_available" == "true" &&
        "$torch_device_count" =~ ^[0-9]+$ &&
        "$torch_device_count" -eq "$EXPECTED_DEVICES" ]]; then
        record torch_npu_devices OK \
            "count=${torch_device_count}; names=${torch_device_names:-unknown}"
    else
        record torch_npu_devices FAIL \
            "available=${npu_available:-unknown}; count=${torch_device_count:-unknown}; expected=${EXPECTED_DEVICES}; ${python_error:-device probe failed}"
    fi
fi

topology_text=""
if command -v npu-smi >/dev/null 2>&1; then
    topology_text="$(run_probe npu-smi info -t topo || true)"
    if [[ -z "$topology_text" ]]; then
        topology_text="$(run_probe npu-smi info -t topology || true)"
    fi
fi
if [[ -n "$topology_text" ]]; then
    record topology OK "$(printf '%s\n' "$topology_text" | limit_lines 100)"
elif [[ "$REQUIRE_TOPO" == "1" ]]; then
    record topology FAIL "npu-smi topology query returned no output (H3_REQUIRE_TOPO=1)"
else
    record topology WARN "npu-smi topology query returned no output (H3_REQUIRE_TOPO=0)"
fi

declare -a hbm_details=()
if command -v npu-smi >/dev/null 2>&1 && ((${#device_paths[@]})); then
    for path in "${device_paths[@]}"; do
        device_id="${path##*/davinci}"
        if [[ ! "$device_id" =~ ^[0-9]+$ ]]; then
            continue
        fi
        memory_output="$(
            run_probe npu-smi info -t memory -i "$device_id" || true
        )"
        if grep -qi 'HBM' <<< "$memory_output"; then
            compact="$(
                printf '%s\n' "$memory_output" |
                    grep -Ei 'HBM|Capacity|Used' |
                    tr '\n' ' '
            )"
            hbm_details+=("npu${device_id}: ${compact% }")
        fi
    done
fi
if ((${#hbm_details[@]})); then
    hbm_detail="$(printf '%s\n' "${hbm_details[@]}" | limit_lines 16)"
    record hbm OK "$hbm_detail"
elif [[ -n "$npu_smi_text" ]] && grep -qi 'HBM' <<< "$npu_smi_text"; then
    hbm_detail="$(
        printf '%s\n' "$npu_smi_text" |
            grep -Ei 'HBM|Memory-Usage' |
            limit_lines 40
    )"
    record hbm FAIL \
        "per-device HBM query unavailable; npu-smi main table follows"$'\n'"${hbm_detail}"
else
    record hbm FAIL "HBM usage not reported by npu-smi"
fi

cpu_model=""
cpu_count=""
if command -v lscpu >/dev/null 2>&1; then
    cpu_model="$(
        LC_ALL=C lscpu |
            awk -F: '/Model name|BIOS Model name/ {
                sub(/^[[:space:]]+/, "", $2);
                sub(/[[:space:]]+$/, "", $2);
                print $2;
                exit
            }'
    )"
fi
if [[ -z "$cpu_model" && -r /proc/cpuinfo ]]; then
    cpu_model="$(
        awk -F: '/^model name|^Model/ {
            sub(/^[[:space:]]+/, "", $2);
            print $2;
            exit
        }' /proc/cpuinfo
    )"
fi
cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
if [[ -z "$cpu_count" && -r /proc/cpuinfo ]]; then
    cpu_count="$(grep -c '^processor' /proc/cpuinfo || true)"
fi
if [[ "$cpu_count" =~ ^[1-9][0-9]*$ ]] &&
    awk -v count="$cpu_count" -v minimum="$MIN_CPU_CORES" \
        'BEGIN { exit !(count >= minimum) }'; then
    record cpu OK "cores=${cpu_count}; model=${cpu_model:-unknown}"
else
    record cpu FAIL \
        "cores=${cpu_count:-unknown}; minimum=${MIN_CPU_CORES}; model=${cpu_model:-unknown}"
fi

ram_total=""
ram_available=""
if [[ -r /proc/meminfo ]]; then
    ram_total="$(awk '/^MemTotal:/ { printf "%.2f", $2 / 1048576 }' /proc/meminfo)"
    ram_available="$(
        awk '/^MemAvailable:/ { printf "%.2f", $2 / 1048576 }' /proc/meminfo
    )"
fi
if [[ -n "$ram_total" ]] &&
    awk -v total="$ram_total" -v minimum="$MIN_RAM_GIB" \
        'BEGIN { exit !(total >= minimum) }'; then
    record ram OK "total_gib=${ram_total}; available_gib=${ram_available:-unknown}"
else
    record ram FAIL \
        "total_gib=${ram_total:-unknown}; minimum_gib=${MIN_RAM_GIB}; available_gib=${ram_available:-unknown}"
fi

disk_check() {
    local key="$1"
    local target="$2"
    local minimum="$3"
    local current="$target"
    local free_kb=""
    local free_gib=""
    local status="OK"
    local detail=""

    while [[ ! -e "$current" ]]; do
        local parent
        parent="$(dirname "$current")"
        if [[ "$parent" == "$current" ]]; then
            break
        fi
        current="$parent"
    done
    if [[ -e "$current" ]]; then
        free_kb="$(df -Pk "$current" 2>/dev/null | awk 'NR == 2 { print $4 }')"
    fi
    if [[ "$free_kb" =~ ^[0-9]+$ ]]; then
        free_gib="$(awk -v kb="$free_kb" 'BEGIN { printf "%.2f", kb / 1048576 }')"
    fi

    if [[ ! -e "$target" ]]; then
        status="WARN"
        detail="${target}=MISSING(free_at=${current}: ${free_gib:-unknown} GiB)"
    else
        detail="${target}: free_gib=${free_gib:-unknown}"
    fi
    if [[ -z "$free_gib" ]]; then
        status="FAIL"
        detail+="; free space unavailable"
    elif ! awk -v free="$free_gib" -v minimum="$minimum" \
        'BEGIN { exit !(free >= minimum) }'; then
        status="FAIL"
        detail+="; minimum_gib=${minimum}"
    fi
    record "$key" "$status" "$detail"
}

disk_check disk_runtime "$RUNTIME_ROOT" "$MIN_RUNTIME_DISK_GIB"
declare -a model_roots=()
IFS=: read -r -a model_roots <<< "$MODEL_ROOT"
model_index=0
for model_root in "${model_roots[@]}"; do
    [[ -z "$model_root" ]] && continue
    model_index=$((model_index + 1))
    disk_check "disk_model_${model_index}" "$model_root" "$MIN_DISK_GIB"
done
if ((model_index == 0)); then
    record disk_model FAIL "H3_MODEL_ROOT contains no paths"
fi

if command -v "$python_bin" >/dev/null 2>&1; then
    if model_output="$(
        "$python_bin" - "$HASH_MODE" "$MANIFEST" "$MODEL_ROOT" <<'PY' 2>&1
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HASH_MODE, MANIFEST_TEXT, ROOTS_TEXT = sys.argv[1:4]
MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


def finish(status: str, summary: str, details: list[str] | None = None) -> None:
    print(f"{status}\t{summary}")
    for line in details or []:
        print(line)
    raise SystemExit(0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid_destination(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "destination must be a non-empty string"
    if "\\" in value or "\x00" in value:
        return None, "destination must use forward slashes and contain no NUL"
    path = Path(value)
    if path.is_absolute():
        return None, "destination must be relative to a model root"
    if any(part in {"", ".", ".."} for part in value.split("/")):
        return None, "destination must not contain empty, '.' or '..' components"
    return value, None


def candidate_files(destination: str, roots: list[Path]) -> tuple[list[Path], list[str]]:
    candidates: list[Path] = []
    errors: list[str] = []
    for root in roots:
        candidate = root / destination
        if not candidate.exists() and not candidate.is_symlink():
            continue
        try:
            resolved_root = root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            errors.append(f"{candidate}: path is outside model root or unreadable: {exc}")
            continue
        if not resolved.is_file():
            errors.append(f"{candidate}: not a regular file")
            continue
        candidates.append(resolved)
    return candidates, errors


def manifest_report(manifest_path: Path, roots: list[Path]) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        finish("FAIL", f"manifest unreadable: {exc}")
    if not isinstance(payload, list) or not payload:
        finish("FAIL", "manifest must be a non-empty JSON list")

    details: list[str] = []
    failures = 0
    warnings = 0
    seen: set[str] = set()
    for index, item in enumerate(payload):
        prefix = f"[{index}]"
        if not isinstance(item, dict):
            details.append(f"{prefix} FAIL: entry must be an object")
            failures += 1
            continue
        destination, destination_error = valid_destination(item.get("destination"))
        if destination_error:
            details.append(f"{prefix} FAIL: {destination_error}")
            failures += 1
            continue
        assert destination is not None
        if destination in seen:
            details.append(f"{prefix} FAIL: duplicate destination {destination}")
            failures += 1
            continue
        seen.add(destination)

        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            details.append(f"{prefix} FAIL: {destination}: size must be positive")
            failures += 1
            continue
        expected_sha = item.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            details.append(
                f"{prefix} FAIL: {destination}: sha256 must be 64 lowercase hex characters"
            )
            failures += 1
            continue

        candidates, path_errors = candidate_files(destination, roots)
        for error in path_errors:
            details.append(f"{prefix} FAIL: {destination}: {error}")
            failures += 1
        if not candidates:
            if not path_errors:
                details.append(
                    f"{prefix} FAIL: {destination}: not found under configured roots"
                )
                failures += 1
            continue
        if len(candidates) > 1:
            warnings += 1
            details.append(
                f"{prefix} WARN: {destination}: found in {len(candidates)} roots; "
                f"using {candidates[0]}"
            )
        target = candidates[0]
        try:
            actual_size = target.stat().st_size
        except OSError as exc:
            details.append(f"{prefix} FAIL: {destination}: stat failed: {exc}")
            failures += 1
            continue
        if actual_size != size:
            details.append(
                f"{prefix} FAIL: {destination}: size={actual_size}; expected={size}"
            )
            failures += 1
            continue
        if HASH_MODE == "none":
            details.append(
                f"{prefix} SKIP: {destination}: size={actual_size}; SHA256 unverified"
            )
            continue
        try:
            actual_sha = sha256_file(target)
        except OSError as exc:
            details.append(
                f"{prefix} FAIL: {destination}: SHA256 failed: {exc}"
            )
            failures += 1
            continue
        if actual_sha != expected_sha:
            details.append(
                f"{prefix} FAIL: {destination}: sha256={actual_sha}; "
                f"expected={expected_sha}"
            )
            failures += 1
            continue
        details.append(
            f"{prefix} OK: {destination}: size={actual_size}; sha256={actual_sha}"
        )

    if failures:
        status = "FAIL"
    elif warnings or HASH_MODE == "none":
        status = "WARN"
    else:
        status = "OK"
    finish(
        status,
        f"manifest={manifest_path}; entries={len(payload)}; "
        f"failures={failures}; warnings={warnings}; hashes={HASH_MODE}",
        details,
    )


def discovery_report(roots: list[Path]) -> None:
    files: set[Path] = set()
    details: list[str] = []
    for root in roots:
        if not root.is_dir():
            details.append(f"ROOT MISSING: {root}")
            continue
        try:
            for directory, _subdirs, filenames in os.walk(root):
                for filename in filenames:
                    candidate = Path(directory) / filename
                    if candidate.suffix.lower() in MODEL_SUFFIXES:
                        files.add(candidate)
        except OSError as exc:
            details.append(f"ROOT FAIL: {root}: {exc}")
    if not files:
        finish(
            "FAIL",
            "NO_EXPECTED; no model-sized files found; SHA256 unverified",
            details,
        )
    total = 0
    hash_failures = 0
    for path in sorted(files):
        try:
            size = path.stat().st_size
        except OSError as exc:
            details.append(f"FAIL: {path}: stat failed: {exc}")
            hash_failures += 1
            continue
        total += size
        if HASH_MODE == "none":
            details.append(f"SIZE: {path}: bytes={size}; SHA256 skipped")
            continue
        try:
            actual_sha = sha256_file(path)
        except OSError as exc:
            details.append(f"FAIL: {path}: SHA256 failed: {exc}")
            hash_failures += 1
            continue
        details.append(f"SHA256: {path}: bytes={size}; sha256={actual_sha}")
        if path.suffix.lower() == ".safetensors":
            details[-1] += "; expected=NO_EXPECTED"
    if hash_failures:
        finish(
            "FAIL",
            f"NO_EXPECTED; model read/hash failures={hash_failures}; "
            f"files={len(files)}; total_bytes={total}",
            details,
        )
    status = "WARN"
    if HASH_MODE == "none":
        verification = "SHA256 skipped by H3_HASH_MODELS=none"
    else:
        verification = "actual SHA256 recorded; no expected digest"
    finish(
        status,
        f"NO_EXPECTED; {verification}; files={len(files)}; "
        f"total_bytes={total}",
        details,
    )


try:
    hash_mode = HASH_MODE
    roots = [
        Path(value).expanduser()
        for value in ROOTS_TEXT.split(os.pathsep)
        if value
    ]
    if MANIFEST_TEXT:
        manifest_report(Path(MANIFEST_TEXT).expanduser(), roots)
    discovery_report(roots)
except SystemExit:
    raise
except Exception as exc:
    finish("FAIL", f"model check crashed: {type(exc).__name__}: {exc}")
PY
    )"; then
        model_header="$(first_line "$model_output")"
        IFS=$'\t' read -r model_status model_summary <<< "$model_header"
        case "$model_status" in
            OK|WARN|FAIL) ;;
            *)
                model_status="FAIL"
                model_summary="model check returned malformed output"
                ;;
        esac
        model_body="$(printf '%s\n' "$model_output" | awk 'NR > 1')"
        model_detail="$model_summary"
        if [[ -n "$model_body" ]]; then
            model_detail+=$'\n'"$model_body"
        fi
        record model_files "$model_status" "$model_detail"
    else
        # The heredoc normally reports failures in-band. This branch covers an
        # interpreter crash that happened before it could do so.
        STATUS["model_files"]="FAIL"
        DETAIL["model_files"]="model checker failed to run"
        FAILURES=$((FAILURES + 1))
    fi
fi

host="$(hostname 2>/dev/null || printf 'unknown')"
generated_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
mapfile -t sorted_keys < <(
    printf '%s\n' "${!STATUS[@]}" | LC_ALL=C sort
)

if [[ "$SCAN_MODE" == "json" ]]; then
    if ((${#sorted_keys[@]} == 0)); then
        sorted_keys=()
    fi
    printf '{\n'
    printf '  "schema": 1,\n'
    printf '  "mode": "read-only",\n'
    printf '  "generated_at": %s,\n' "$(json_string "$generated_at")"
    printf '  "hostname": %s,\n' "$(json_string "$host")"
    printf '  "ok": %s,\n' "$([[ "$FAILURES" -eq 0 ]] && printf true || printf false)"
    printf '  "failures": %d,\n' "$FAILURES"
    printf '  "warnings": %d,\n' "$WARNINGS"
    printf '  "checks": [\n'
    for index in "${!sorted_keys[@]}"; do
        key="${sorted_keys[$index]}"
        separator=","
        if ((index == ${#sorted_keys[@]} - 1)); then
            separator=""
        fi
        printf '    {"key": %s, "status": %s, "detail": %s}%s\n' \
            "$(json_string "$key")" \
            "$(json_string "${STATUS[$key]}")" \
            "$(json_string "${DETAIL[$key]}")" \
            "$separator"
    done
    printf '  ]\n'
    printf '}\n'
else
    printf '%-22s %-5s %s\n' CHECK STATUS DETAIL
    for key in "${sorted_keys[@]}"; do
        detail="${DETAIL[$key]}"
        detail="${detail//$'\n'/$'\n'                         }"
        printf '%-22s %-5s %s\n' "$key" "${STATUS[$key]}" "$detail"
    done
    printf '\nRESULT %s failures=%d warnings=%d host=%s\n' \
        "$([[ "$FAILURES" -eq 0 ]] && printf PASS || printf FAIL)" \
        "$FAILURES" "$WARNINGS" "$host"
fi

[[ "$FAILURES" -eq 0 ]]
