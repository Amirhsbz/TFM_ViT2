# 数据校对
## 左右夹爪互换
在dp部署时，增加参数--swap-tactile-lr-for-policy，只会影响策略输入，而不会影响相机显示和marker tracking

在pi0部署时，在client端增加参数--swap-tactile-lr-for-policy，效果相同

在用marker tracking,也就是使用--enable-marker-tracking的情况下也兼容

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
  shared/data/bc_data/put_bottle_upright \
  --summary-json shared/data/bc_data/put_bottle_upright/contact_gate_summary.json \
  --write-contact-segments （可选）
会使用h5文件里已有的frames/tactile_left_marker_motion和frames/tactile_right_marker_motion变量，计算并新增frames/contact_marker_motion（左右的max），frames/contact_gate（每一帧的接触标记，0等待固定触觉图像替换，1用实时触觉图像），contact_gate_config attr变量，而统计信息保存在contact_gate_summary.json里
默认情况下，frames/contact_marker_motion保存的是左右 tactile motion 各自扣掉前20帧median baseline后的max delta；如果要恢复旧的绝对motion逻辑，可以加 --motion-baseline-mode none
如果是多阶段任务，需要加write-contact-segments，让每次夹爪松开都使用最近的帧来计算marker motion baseline；这会新增frames/contact_gate_segment_id：contact_gate=1时为-1，每一段contact_gate=0分别编号为0、1、2...。后续生成多阶段baseline数据集时会优先使用这个变量；如果没有写入，生成脚本也会根据contact_gate自动推断。

先 dry-run 看阈值是否合适，不写入原数据，可以先跑：
python3 Data_analysis/compute_tactile_contact_gate.py \
  shared/data/bc_data/peg_in_hole \
  --dry-run  \
  --write-contact-segments （可选）
这个模式会在终端打印每条轨迹的统计信息，最后打印一个 JSON summary
  "mean_contact_ratio": 0.6736326334618639, 平均每条轨迹约 67% 的时间 gate=1，也就是使用实时触觉。
  "median_first_contact_frame": 91.5, 一半 episode 的 gate 大约在第 92 帧打开。若视频是 15 FPS，就是约 6.1s，把这个时间和视频的实际时间对比看看是否合适；
如果这个任务的接触时间明显偏早/偏晚，可以调节这些参数，直接加在compute_tactile_contact_gate.py运行命令后：
    "contact_on_threshold": 0.5, 根据marker motion设置contact_gate =1的阈值，实际marker motion大于它才可能视为有接触；如果实时触觉开太早，把它调高
    "contact_off_threshold": 0.2, 如果 gate 中间容易断，把它调低
    "contact_on_consecutive_frames": 2,
    "contact_off_consecutive_frames": 8, 如果 gate 中间容易断，把它调高
    "gripper_on_threshold": 0.2, 夹爪往里闭合的距离阈值，大于这个值才允许 contact_gate =1（判定为有接触）发生
    "gripper_open_threshold": 0.08, 允许 gate contact_gate=0（判定为松开），如果 gate 保持太久、松开后还不关，把它调高（也就是夹爪闭合距离挺大的时候就算松开了）
    "gripper_closed_threshold": 0.2,
    "gripper_hold_threshold": 0.2,
    "gripper_open_consecutive_frames": 2, 如果出现闪烁，把它调大
    "hold_contact_while_gripper_closed": true,
    "motion_baseline_mode": "early_median", 在这个模式下，用motion-baseline-num-frames帧的图像（默认20）来计算基线marker motion,后续的contact_gate 不是看原始 motion，而是看“比开头静止状态多出来多少”。
    "motion_baseline_num_frames": 20
如果希望恢复“marker motion低就可以关gate”的旧逻辑，可以额外加 --disable-gripper-hold

2）生成 baseline-gated tactile 数据集：
python3 Data_analysis/make_contact_gated_tactile_dataset.py \
  shared/data/bc_data/put_bottle_upright \
  shared/data/bc_data/put_bottle_upright_gated_tactile \
  --overwrite \
  --mode segment-baseline （可选）
根据h5文件的contact_gate值，决定哪些触觉图像保留，哪些替换，并完成替换，生成新数据集wipe_board_gated_tactile；
加上 --overwrite 后，如果 wipe_board_gated_tactile 已经存在，脚本会删除、替换或重新生成其中的数据；
mode segment-baseline：如果松开夹爪后传感器形变状态和episode开头不同，或者任务需要松开后重新交互，可以用多阶段固定图像。它仍然只替换contact_gate=0的触觉图像，但每一段contact_gate=0都会用该段最早的baseline-num-frames帧重新计算固定图像，而不是一直使用episode开头的固定图像。

如果想更直观检查 contact_gate 的效果，可以生成黑帧检查版数据集：
python3 Data_analysis/make_contact_gated_tactile_dataset.py \
  shared/data/bc_data/peg_in_hole \
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
  shared/data/bc_data/put_bottle_upright_gated_tactile/0518_165953/trajectory.h5 \
  --combined-only
