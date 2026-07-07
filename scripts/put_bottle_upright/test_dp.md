
python run_env.py \
  --agent dp \
  --dp-ckpt-path data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast/0428_205543_RIzG-camera=01-identity=False-repr=IP-oh=2-ah=4-ph=12-prefix=None-do=0.0-imgos=64-wd=1e-05-use_ddim=False-binarize_touch=False-posdelta/last.ckpt \
  --save-data \
  --swap-tactile-lr-for-policy

如果 DP 是用 marker tracking overlay 版触觉视频训练的，例如
`shared/data/bc_data/put_bottle_upright_tactile_crop` 中的
`/videos/tactile_left_rgb` 和 `/videos/tactile_right_rgb` 已经被替换成带箭头版本，
测试时需要让实时 marker tracking 结果作为 policy 输入：

如果当前相机位置或分辨率和采集训练数据时不一致，先用当前触觉相机画面重新选测试裁剪框：

python Data_analysis/select_tactile_crop_from_cameras.py \
  --config-dir sensor_configs/put_bottle_upright_test \
  --output-size 320x240 \
  --overwrite

按顺序点击：左上、右上、右下、左下。这个配置是直接基于当前实时相机帧的，
测试时会作为触觉相机唯一的透视裁剪配置使用。

python run_env.py \
  --agent dp \
  --dp-ckpt-path <your_marker_tracking_dp_ckpt> \
--swap-tactile-lr-for-policy \
  --use-tactile \
  --enable-marker-tracking \
  --tactile-crop-config-dir sensor_configs/put_bottle_upright_test \
  --use-marker-tracking-overlay-for-policy

`--tactile-crop-config-dir` 会让实时触觉相机直接输出该裁剪后的图像，
marker tracking 也会画在同一张触觉图像上；不会再额外做第二次裁剪。
`tactile_left_marker_tracking` / `tactile_right_marker_tracking` 窗口显示的就是
policy 实际看到的 marker tracking overlay。

程序启动后会先把机器人移动到 reset joints，然后提示：
- 按一下并松开键盘 r：移动到初始位置并开始执行 policy
- 双击 r：停止并保存当前 trajectory
- 三击 r：停止并删除当前 trajectory
- Ctrl+C：退出程序