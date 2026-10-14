#!/bin/bash

case "${INSTALL_TYPE:-}" in
    pip|conda) ;;
    *)
        printf 'Unsupported INSTALL_TYPE: %s. Supported modes: pip, conda.\n' \
            "${INSTALL_TYPE:-<unset>}" >&2
        exit 1
        ;;
esac

echo "Activate virtual environment"
source ci/activate_env.sh

set -ex

PIPI="pip install --timeout=60"

if [ "$USE_PRE" == "1" ] || [ "$USE_PRE" == true ]; then
    PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre";
fi

#---------- FURY Installation -----------------

$PIPI .

set +ex
