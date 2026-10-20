#!/bin/bash
set -eo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${ROOT}/patches/modeling_llama.py"
TARGET="$(python -c 'import os, transformers; print(os.path.join(os.path.dirname(transformers.__file__), "models", "llama", "modeling_llama.py"))' | tail -n 1)"
VERSION="$(python -c 'import transformers; print(transformers.__version__)' | tail -n 1)"
MODE="${1:-link}"

check() {
    python -c 'from transformers.models.llama import modeling_llama as m; import sys; sys.exit(0 if getattr(m, "BIFOCAL_PATCH", False) else 1)' \
        && echo "Bifocal patch is active: ${TARGET}" \
        || { echo "Bifocal patch is NOT active: ${TARGET}"; exit 1; }
}

case "${MODE}" in
    check)
        check
        ;;
    restore)
        [ -L "${TARGET}" ] && rm -f "${TARGET}"
        cp "${ROOT}/patches/modeling_llama_orig.py" "${TARGET}"
        echo "Restored the original transformers ${VERSION} modeling_llama.py"
        ;;
    link|copy)
        [ "${VERSION}" = "4.54.0" ] || { echo "transformers==4.54.0 is required, found ${VERSION}"; exit 1; }
        rm -f "${TARGET}"
        if [ "${MODE}" = "copy" ]; then cp "${SOURCE}" "${TARGET}"; else ln -s "${SOURCE}" "${TARGET}"; fi
        check
        ;;
    *)
        echo "usage: bash install.sh [link|copy|check|restore]"
        exit 2
        ;;
esac
