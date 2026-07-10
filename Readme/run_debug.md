# 数据校对
## 左右夹爪互换

## 只在接触时提供触觉信息
当夹爪正在移动并接近物体的时候，只用视觉信息，而在夹爪与物体发生接触后，再把触觉信息加入进去
训练和测试时网络结构保持一致，触觉相机一直采集，但在接触前不给 policy 使用；接触后再把真实 tactile 输入放开。
定义一个 contact_gate，夹爪接触判定可以先用现有 marker motion。离线数据里不同 tactile stream 的预接触 motion 基线可能不一样，所以先对左右 tactile motion 各自减去 episode 前 20 帧 median baseline，再取左右 delta motion 的 max，逻辑是：
'''
baseline-corrected marker motion > 0.5 连续 2 帧
AND
gripper_position >= 0.2
'''
用 hysteresis，避免 gate 抖动。gate打开之后，如果gripper stroke仍处于闭合/保持区间（默认 gripper_position >= 0.2），不会因为marker motion短暂下降就关掉gate。
夹爪松开判定：
'''
baseline-corrected marker motion < 0.2 连续 8 帧，且gripper不再处于闭合/保持区间
OR
接触后夹爪曾经闭合过，然后 gripper_position <= 0.08 连续 2 帧
'''
训练时，对每一帧做同样的 gating，用每个episode 前 20 个 pre-contact tactile 帧的 median baseline作为固定触觉图像，在不提供实时触觉的时候用它来替换

核心代码：
learning/tactile_contact_gate.py
Data_analysis/compute_tactile_contact_gate.py

数据集处理：
把接触前和松开后的触觉传感器图像用固定图像替换，保存在shared/data/bc_data/*_gated_tactile内

可复用模块：

1）给原始 H5 计算 contact_gate：
python3 Data_analysis/compute_tactile_contact_gate.py \
  shared/data/bc_data/wipe_board \
  --summary-json shared/data/bc_data/wipe_board/contact_gate_summary.json
会使用h5文件里已有的frames/tactile_left_marker_motion和frames/tactile_right_marker_motion变量，计算并新增frames/contact_marker_motion（左右的max），frames/contact_gate（每一帧的接触标记，0等待固定触觉图像替换，1用实时触觉图像），contact_gate_config attr变量，而统计信息保存在contact_gate_summary.json里
默认情况下，frames/contact_marker_motion保存的是左右 tactile motion 各自扣掉前20帧median baseline后的max delta；如果要恢复旧的绝对motion逻辑，可以加 --motion-baseline-mode none

先 dry-run 看阈值是否合适，不写入原数据，可以先跑：
python3 Data_analysis/compute_tactile_contact_gate.py \
  shared/data/bc_data/turn_cleanser_water_bottle \
  --dry-run
这个模式会在终端打印每条轨迹的统计信息，最后打印一个 JSON summary
如果这个任务的接触时间明显偏早/偏晚，可以调节这些参数，直接加在compute_tactile_contact_gate.py运行命令后：
--contact-on-threshold 0.5
--contact-off-threshold 0.2
--contact-off-consecutive-frames 8
--gripper-on-threshold 0.2
--gripper-hold-threshold 0.2
--gripper-open-threshold 0.08
--motion-baseline-num-frames 20
如果希望恢复“marker motion低就可以关gate”的旧逻辑，可以额外加 --disable-gripper-hold

2）生成 baseline-gated tactile 数据集：
python3 Data_analysis/make_contact_gated_tactile_dataset.py \
  shared/data/bc_data/wipe_board \
  shared/data/bc_data/wipe_board_gated_tactile \
  --overwrite
根据h5文件的contact_gate值，决定哪些触觉图像保留，哪些替换，并完成替换，生成新数据集wipe_board_gated_tactile；
加上 --overwrite 后，如果 wipe_board_gated_tactile 已经存在，脚本会删除、替换或重新生成其中的数据

如果想更直观检查 contact_gate 的效果，可以生成黑帧检查版数据集：
python3 Data_analysis/make_contact_gated_tactile_dataset.py \
  shared/data/bc_data/wipe_board \
  --mode black \
  --overwrite
这个模式会自动保存到 shared/data/bc_data/wipe_board_gated_black，把 contact_gate=0 的触觉帧替换为全黑图像，contact_gate=1 的帧保留真实触觉图像；
也可以显式指定输出目录：
python3 Data_analysis/make_contact_gated_tactile_dataset.py \
  shared/data/bc_data/wipe_board \
  shared/data/bc_data/wipe_board_gated_black \
  --mode black \
  --overwrite

3）导出处理后的视频检查：
python3 Data_analysis/test_h5_video_export.py \
  shared/data/bc_data/wipe_board_gated_tactile/0428_155943/trajectory.h5 \
  --combined-only
