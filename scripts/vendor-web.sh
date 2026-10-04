#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 John Luke NIKABOU (LucNIK)
#
# Third-party files the site serves itself (no CDN, no third-party request at runtime):
# IBM Plex fonts and MapLibre GL JS, at pinned versions, copied into web/.
set -euo pipefail
MAPLIBRE_VERSION="5.6.0"
WEB="$(cd "$(dirname "$0")/../web" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

mkdir -p "$WEB/assets/fonts" "$WEB/vendor"
npm pack --silent @fontsource/ibm-plex-sans@5 @fontsource/ibm-plex-mono@5 >/dev/null
for f in fontsource-ibm-plex-*.tgz; do
  tar -xzf "$f"
  cp package/files/*-latin-{400,500,600}-normal.woff2 "$WEB/assets/fonts/" 2>/dev/null || true
  rm -rf package
done

npm pack --silent "maplibre-gl@$MAPLIBRE_VERSION" >/dev/null
tar -xzf "maplibre-gl-$MAPLIBRE_VERSION.tgz"
cp package/dist/maplibre-gl.js package/dist/maplibre-gl.css "$WEB/vendor/"
cp package/LICENSE.txt "$WEB/vendor/MAPLIBRE-LICENSE.txt"
echo "$MAPLIBRE_VERSION" > "$WEB/vendor/MAPLIBRE-VERSION"
ls -la "$WEB/assets/fonts" "$WEB/vendor"
