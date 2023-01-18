#!/bin/bash

# Echo Routing mavlink message
echo "[SITL] Routing mavlink message"

# Get the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Append router_sitl.conf to the current directory
CONF="$DIR/router_hexa.conf"

# Execute mavlink-routerd -c CONF
mavlink-routerd -c $CONF
