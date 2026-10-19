#!/bin/bash
# Helper script to run ROS 2 commands and colcon builds inside the Docker development container.
# Host: Ubuntu 26.04 (Noble/Jazzy runs inside container)

set -euo pipefail

IMAGE_NAME="eeazybot-dev:latest"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# Ensure image exists or build from .devcontainer/Dockerfile
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Image '$IMAGE_NAME' not found. Building from .devcontainer/Dockerfile..."
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/.devcontainer/Dockerfile" "$SCRIPT_DIR/.devcontainer"
fi

DOCKER_ARGS=(
    --rm
    -i
    --network=host
    -v "$SCRIPT_DIR:/workspaces"
    -w /workspaces
    -u "$(id -u):$(id -g)"
    -e "DISPLAY=${DISPLAY:-}"
    -e "QT_X11_NO_MITSHM=1"
)

# Load environment variables from .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    DOCKER_ARGS+=(--env-file "$SCRIPT_DIR/.env")
fi

# Allocate TTY if connected to a terminal
if [ -t 0 ]; then
    DOCKER_ARGS+=(-t)
fi

# X11 GUI forwarding if socket exists
if [ -d "/tmp/.X11-unix" ]; then
    DOCKER_ARGS+=(-v "/tmp/.X11-unix:/tmp/.X11-unix:rw")
fi

# Hardware device access
if [ -d "/dev" ]; then
    DOCKER_ARGS+=(--privileged -v "/dev:/dev")
fi

# Shortcut commands
CMD_ARGS=()
case "${1:-}" in
    ""|shell)
        exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" bash
        ;;
    build)
        shift
        CMD_ARGS=(colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@")
        ;;
    test)
        shift
        CMD_ARGS=(colcon test "$@")
        ;;
    *)
        CMD_ARGS=("$@")
        ;;
esac

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" bash -c \
    'source /opt/ros/jazzy/setup.bash && if [ -f /workspaces/install/setup.bash ]; then source /workspaces/install/setup.bash; fi && exec "$@"' _ "${CMD_ARGS[@]}"
