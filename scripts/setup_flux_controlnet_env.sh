#!/bin/bash
set -euo pipefail

DIFFUSERS_DIR=${1:?usage: setup_flux_controlnet_env.sh DIFFUSERS_DIR VENV_DIR [REF] [BASE_ENV_PYTHON]}
VENV_DIR=${2:?usage: setup_flux_controlnet_env.sh DIFFUSERS_DIR VENV_DIR [REF] [BASE_ENV_PYTHON]}
DIFFUSERS_REF=${3:-main}
BASE_ENV_PYTHON=${4:-}
SENTINEL="${VENV_DIR}/.flux_controlnet_env_ready"
META_FILE="${VENV_DIR}/.flux_controlnet_env_meta"

if [ ! -d "${DIFFUSERS_DIR}/.git" ]; then
    git clone https://github.com/huggingface/diffusers.git "${DIFFUSERS_DIR}"
fi

git -C "${DIFFUSERS_DIR}" fetch origin
if git -C "${DIFFUSERS_DIR}" show-ref --verify --quiet "refs/remotes/origin/${DIFFUSERS_REF}"; then
    git -C "${DIFFUSERS_DIR}" checkout -B "${DIFFUSERS_REF}" "origin/${DIFFUSERS_REF}"
else
    git -C "${DIFFUSERS_DIR}" checkout "${DIFFUSERS_REF}"
fi

DIFFUSERS_REV=$(git -C "${DIFFUSERS_DIR}" rev-parse HEAD)
REQUIREMENTS_HASH=$(sha256sum "${DIFFUSERS_DIR}/examples/controlnet/requirements_flux.txt" | awk '{print $1}')
SETUP_HASH=$(sha256sum "$0" | awk '{print $1}')
BASE_PYTHON_VERSION=""
if [ -n "${BASE_ENV_PYTHON}" ]; then
    BASE_PYTHON_VERSION=$("${BASE_ENV_PYTHON}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
fi

EXPECTED_META=$(cat <<EOF
ref=${DIFFUSERS_REF}
rev=${DIFFUSERS_REV}
requirements_sha256=${REQUIREMENTS_HASH}
setup_sha256=${SETUP_HASH}
base_python=${BASE_ENV_PYTHON}
base_python_version=${BASE_PYTHON_VERSION}
EOF
)

if [ -f "${SENTINEL}" ] && [ -f "${META_FILE}" ] && [ "$(cat "${META_FILE}")" = "${EXPECTED_META}" ]; then
    echo "FLUX ControlNet environment already prepared at ${VENV_DIR}"
    exit 0
fi

rm -rf "${VENV_DIR}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    if [ -n "${BASE_ENV_PYTHON}" ]; then
        "${BASE_ENV_PYTHON}" -m venv "${VENV_DIR}"
    else
        python3 -m venv "${VENV_DIR}"
    fi
fi

if [ -n "${BASE_ENV_PYTHON}" ]; then
    BASE_SITE_PACKAGES=$("${BASE_ENV_PYTHON}" -c 'import site; print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))')
    TARGET_SITE_PACKAGES=$("${VENV_DIR}/bin/python" -c 'import site; print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))')
    printf '%s\n' "${BASE_SITE_PACKAGES}" > "${TARGET_SITE_PACKAGES}/_base_env_overlay.pth"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${DIFFUSERS_DIR}"
python -m pip install -r "${DIFFUSERS_DIR}/examples/controlnet/requirements_flux.txt"
python -m pip install peft sentencepiece protobuf

printf '%s\n' "${EXPECTED_META}" > "${META_FILE}"
touch "${SENTINEL}"
echo "Prepared FLUX ControlNet environment at ${VENV_DIR}"
