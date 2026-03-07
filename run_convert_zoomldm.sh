#!/bin/bash
# Convert StonyBrook-CVLab/ZoomLDM to diffusers format and save to BiliSakura/ZoomLDM
set -e
cd "$(dirname "$0")"
conda activate rsgen 2>/dev/null || true
python convert_to_diffusers.py \
    --repo_id StonyBrook-CVLab/ZoomLDM \
    --variant brca \
    --output_dir /root/worksapce/models/BiliSakura/ZoomLDM
