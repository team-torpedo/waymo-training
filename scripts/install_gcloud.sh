#!/usr/bin/env bash
# Install the Google Cloud SDK (gcloud + gsutil) needed to download the Waymo dataset.
# Linux/macOS, installs into ~/google-cloud-sdk. See https://cloud.google.com/sdk/docs/install
set -euo pipefail

if command -v gsutil >/dev/null 2>&1; then
  echo "gsutil already installed: $(command -v gsutil)"
else
  echo ">> Installing the Google Cloud SDK via the official installer"
  curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
  echo ">> Restart your shell (or run: source ~/google-cloud-sdk/path.bash.inc)"
fi

cat <<'EOF'

Next steps (once, interactive):
  1. Register and accept the license at https://waymo.com/open/ using a Google account.
  2. gcloud auth login
  3. gcloud auth application-default login
  4. Check access:  gsutil ls gs://waymo_open_dataset_v_2_0_0/training/camera_image | head
EOF
