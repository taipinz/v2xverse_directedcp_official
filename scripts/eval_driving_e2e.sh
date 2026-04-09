#!/bin/bash

# $1, route id
# $2, Carla port
# $3, exp_name
# $4, repeat
# $5, agent config
# $6, scenario config

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_python_candidate() {
    local candidate="$1"
    if [ -z "$candidate" ]; then
        return 1
    fi

    if [[ "$candidate" == */* ]]; then
        [ -x "$candidate" ] || return 1
        printf '%s\n' "$candidate"
        return 0
    fi

    command -v "$candidate" 2>/dev/null
}

validate_closed_loop_python() {
    local pybin="$1"
    "$pybin" "${ROOT_DIR}/scripts/check_closed_loop_env.py" --quiet >/dev/null 2>&1
}

select_closed_loop_python() {
    local candidates=()
    if [ -n "${V2XVERSE_PYTHON:-}" ]; then
        candidates+=("${V2XVERSE_PYTHON}")
    fi
    candidates+=("python")
    candidates+=("${HOME}/.local/share/mamba/envs/v2xverse/bin/python")
    candidates+=("${HOME}/miniconda3/envs/v2xverse/bin/python")

    local resolved=""
    local candidate=""
    for candidate in "${candidates[@]}"; do
        resolved="$(resolve_python_candidate "$candidate")" || continue
        if validate_closed_loop_python "$resolved"; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done

    resolved="$(resolve_python_candidate "${V2XVERSE_PYTHON:-python}")" || return 1
    printf '%s\n' "$resolved"
    return 0
}

PYTHON_BIN="$(select_closed_loop_python)"
if [ -z "${PYTHON_BIN}" ]; then
    echo "[closed-loop] unable to find a usable Python interpreter."
    echo "[closed-loop] Set V2XVERSE_PYTHON to a Python 3.7 interpreter with CARLA and simulation dependencies installed."
    exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_closed_loop_env.py"
if [ $? -ne 0 ]; then
    exit 1
fi

PY_TAG="$(${PYTHON_BIN} -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')"
CARLA_EGG="$(find "${ROOT_DIR}/external_paths/carla_root/PythonAPI/carla/dist" -maxdepth 1 -type f -name "carla-*-${PY_TAG}-linux-x86_64.egg" | sort | head -n 1)"
if [ -z "${CARLA_EGG}" ]; then
    echo "[closed-loop] no CARLA egg matches ${PY_TAG} under external_paths/carla_root/PythonAPI/carla/dist"
    exit 1
fi

export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner
export DATA_ROOT=external_paths/data_root
export SAVE_DIR=results

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_EGG}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}/team_code
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=${2:-40000} # IMPORTANT: same as the carla server port
export TM_PORT=`expr $PORT + 5` # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export TRAFFIC_SEED=2000
export CARLA_SEED=2000
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=${LEADERBOARD_ROOT}/data/evaluation_routes/town05_short_r${1:-0}.xml
# verify the evaluation route, including start point and end point.
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios_2.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter$6.yaml
export RESULT_ROOT=${SAVE_DIR}/results_driving_${3:-debug}
export EVAL_SETTING=v2x_final/town05_short_collab/r${1:-0}_repeat${4:-0}
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json
# path to save the result json file
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
# path to save the images.

export TEAM_AGENT=simulation/leaderboard/team_code/pnp_agent_e2e.py
# V2X agent with BEV input to indicate the drivable area.
export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/pnp_config_$5.yaml
# model config file!

export RESUME=0
export EGO_NUM=1
export SKIP_EXISTED=1

mkdir -p $SAVE_PATH
mkdir -p ${RESULT_ROOT}/${EVAL_SETTING}

"${PYTHON_BIN}" ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_parameter.py \
--scenarios=${SCENARIOS}  \
--scenario_parameter=${SCENARIOS_PARAMETER}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--carlaProviderSeed=${CARLA_SEED} \
--trafficManagerSeed=${TRAFFIC_SEED} \
--ego-num=${EGO_NUM} \
--timeout 600 \
--skip_existed=${SKIP_EXISTED}
