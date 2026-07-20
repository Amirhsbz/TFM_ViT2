本机运行（修改参数）：
python run_env.py \
  --agent pi0 \
  --pi0-policy-host 127.0.0.1 \
  --pi0-policy-port 8000 \
  --pi0-prompt "put the bottle upright" \
  --pi0-include-tactile \
  --pi0-tactile-feature-mode image_embedding \
  --use-tactile \
  --enable-marker-tracking \
  --tactile-crop-config-dir sensor_configs/put_bottle_upright_test \
  --use-marker-tracking-overlay-for-policy \
  --swap-tactile-lr-for-policy