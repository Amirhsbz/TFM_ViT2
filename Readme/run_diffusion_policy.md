#TODO
把带marker tracking的测试代码已经写好了，跑的时候加参数，把改的代码同步到amir电脑

# 裁剪触觉图像
- 鼠标选点，生成角点坐标：python Data_analysis/crop_tactile_h5_videos.py select-config shared/data/bc_data/wipe_board --config-dir sensor_configs/wipe_board --output-size 320x240
- 预览裁剪效果：python Data_analysis/crop_tactile_h5_videos.py preview shared/data/bc_data/wipe_board --config-dir sensor_configs/wipe_board
- 批量转换整个数据集：python Data_analysis/crop_tactile_h5_videos.py convert shared/data/bc_data/wipe_board shared/data/bc_data/wipe_board_tactile_crop --config-dir sensor_configs/wipe_board
- 查看裁剪后的视频：python Data_analysis/test_h5_video_export.py shared/data/bc_data/wipe_board_tactile_crop/0428_161030/trajectory.h5
- 做marker tracking
  - 单个数据：python Data_analysis/export_marker_tracking_overlay.py shared/data/bc_data/wipe_board_tactile_crop/0428_162501/trajectory.h5
  - 整个数据集的触觉视频替换成带marker tracking箭头的： python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py shared/data/bc_data/wipe_board_tactile_crop

# 总体流程
- 数据从Amir电脑传到onedrive, 再传到这台电脑
- 检查时间戳一致性：python Data_analysis/check_bc_data_integrity.py shared/data/bc_data/rubiks_cube
- 裁剪首尾静止部分：python Data_analysis/trim_bc_data_by_eef_motion.py \
  shared/data/bc_data/wipe_board \
  shared/data/bc_data/wipe_board_trimmed
- 对裁剪后的数据，检查时间戳一致性：python Data_analysis/check_bc_data_integrity.py shared/data/bc_data/wipe_board_trimmed
- 导出查看裁剪后的视频（抽样）：python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/wipe_board_trimmed
- 把shared/data/bc_data/wipe_board_trimmed数据上传到服务器
- 查看计算资源：
  - cd /scratch/grp/luo/shiyi/project/tele-gsy/scripts
  - ./check_gpu_partition_resources.sh
- 查看存储占用：
  - 查看当前目录"所在磁盘"的总空间和剩余空间（Hpc无效）：df -h /path
  - 查看当前目录的实际占用空间：du -BG -d 1 ./* | sort -n
- 运行准备代码（先修改参数，然后把sh文件上传到服务器）：
  - sbatch -p gpu push_button/run_split_data.sh: 会在data_split文件夹下生成train/test文件夹，里面是指向数据的链接
  - sbatch -p gpu push_button/run_prepare_cache.sh: 每个train/test文件夹都会有一个很大的dat文件，使用默认名称01-False-mem.dat
  - 如果需要tactile图像： sbatch -p gpu push_button/run_prepare_cache_tactile.sh，名称在sh文件中指定
  - pi0：sbatch -p gpu fold_Tshirt/run_convert_pi0_lerobot.sh
- 运行训练代码（先修改参数，然后把sh文件上传到服务器）：sbatch -p gpu push_button/run_train_dp.sh
- 运行后马上看一下out和err输出，记录wandb链接，gpu型号，训练用时，失败原因
- 运行结束后把最后的模型下载下来，压缩，传到网盘：/data/wipe_board_trimmed/ckpts
- 在机器人上运行测试
- 把测试save_data的数据传到onedrive,保存到自己电脑
- 在自己电脑导出运行视频：
  - python Data_analysis/test_h5_video_export.py
  - python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/eval_wipe_board
- 在自己电脑查看运行情况，记录到xlsx

# Additional requirements
pip install h5py
pip install wandb

# Use HPC and run codes
- 连接hpc：ssh -J k25070928@bastion.er.kcl.ac.uk k25070928@hpc.create.kcl.ac.uk
- 查看资源：sinfo
- 查看节点描述：scontrol show node erc-hpc-comp192
- 筛选可用节点：./check_gpu_partition_resources.sh
- 测试GPU调用：/users/k25070928/project/InitialTest/run_torch_test.sh
- 打开交互式窗口：srun -p interruptible_gpu -w erc-hpc-comp054 --pty /bin/bash -l，输入exit退出
- 提交任务：
  - 提交（在sh里写了gpu申请的话，就不用在这里写清楚了，这里的gpu是节点的名字）：sbatch -p gpu -w erc-hpc-comp034 run_preparation.sh，其中的-w erc-hpc-comp034是可选的，用来指定具体的资源名称，避免节点重名搞不清楚
  - 撤回提交：scancel 123456
  - 查看排队：squeue -j 33470026
  - 查看执行情况：sacct -j 33470026
  - 查看输出：cat /scratch/grp/luo/shiyi/project/tele-gsy/script_results/train_dp_33589285.out
- 单独占用窗口（学校的交互式窗口只能用4小时）：srun -p luo_gpu --cpus-per-task 64 --gres gpu --constraint a100 --time=30-00:00:00 --pty /bin/bash -l
- 查看训练过程：
  - 项目文件夹：/scratch/grp/luo/shiyi/project/tele-gsy
  - 提交job的out文件：/scratch/grp/luo/shiyi/project/tele-gsy/script_results/train_dp_33589285.out

# Training
## split data

python workflow/split_data.py \
  --base_path shared/data/bc_data\
  --output_path data_split \
  --data_name rubiks_cube_trimmed \
  --num_trajs 15

解释：它主要是创建软链接 symlink，不是复制完整数据

  --base_path 原始轨迹数据目录的父目录，脚本会实际读取：base_path/grab_03 \
  --output_path ~/yongqiang/tele-gsy/data_split \
  --data_name grab_03 \
  --num_trajs ...  指定从训练集中再抽取多少条轨迹，生成更小的训练子集。它支持多个数字，例如：--num_trajs 10 25 50 会额外生成指定数量的training set：data_split/grab_03_train_10, data_split/grab_03_train_25, data_split/grab_03_train_50

## prepare cache data
参数保存在data/rubiks_cube_trimmed/ckpts/cache_prepare_dummy，在data_split/rubiks_cube_trimmed_train里面生成data_split/rubiks_cube_trimmed_train/01-False-mem.dat

python learning/dp/pipeline.py \
  --data_path /users/k25070928/project/tele-gsy/data_split/rubiks_cube_trimmed \
  --model_save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/ckpts/cache_prepare_dummy \
  --use_train_test_split True \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 \
  --num_workers 4 \
  --obs_horizon 2 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --num_diffusion_iters 100 \
  --use_memmap_cache True \
  --load_img False \
  --gpu 0 \
  --prepare_cache_only True

解释：

pipeline.py: 训练/评估/缓存准备入口脚本 \
  --data_path ~/yongqiang/tele-gsy/data_split/grab_03 数据集路径前缀 \
  --model_save_path /media/rpl/HDD/yongqiang/TactileFundationModel/tele-gsy/data/grab_03/ckpts/cache_prepare_dummy 模型/配置保存路径。即使这里只准备缓存，代码仍会创建一个带时间戳和参数后缀的目录，并保存运行参数。\
  --use_train_test_split True 使用已经切好的 train/test 数据目录，代码实际会读取data_split/grab_03_train和data_split/grab_03_test \
  --representation_type img-pos 输入观测模态，用 - 分割，所以这里等价于：["img", "pos"]，相机 RGB 图像 + joint position\
  --camera_indices 01 使用第 0 和第 1 个相机\
  --joint_state_dim 7 # pos 观测的维度是 7\
  --action_dim 7 # 每个时间步预测 7 维 action\
  --eef_dim 6 # 末端执行器状态 eef 的维度是 6。不过这条命令的 representation_type 是 img-pos，没有用 eef，所以这个参数基本不会参与当前模型输入。\
  --batch_size 32 训练 dataloader 的 batch size。虽然这里不训练，但准备 cache 时会构造 train loader，因此仍会用到\
  --num_workers 4 PyTorch DataLoader 的并行 worker 数量。越大通常读数据越快，但 CPU/内存压力也更高。\
  --obs_horizon 1 模型每次看多少帧历史观测。这里是 1，表示只看当前帧。\
  --pred_horizon 16 模型一次预测多长的动作序列。这里是 16 个未来时间步。\
  --action_horizon 8 实际执行/使用预测动作序列中的多少步\
  --num_diffusion_iters 100 扩散模型采样/去噪步数\
  --use_memmap_cache True 启用图像 memmap 缓存。代码会在 train/test 数据目录下准备类似这样的文件：grab_03_train/01-False-mem.dat，grab_03_test/01-False-mem.dat \
  --load_img False 不把所有图片一次性加载进内存。
在当前设置下，图片会通过 memmap 或文件按需读取。配合 --use_memmap_cache True，通常是更省内存的做法。\
  --gpu 0 使用第 0 张 GPU\
  --prepare_cache_only True 不会真正训练模型，缓存准备完就退出

## train
python learning/dp/pipeline.py \
  --data_path /users/k25070928/project/tele-gsy/data_split/rubiks_cube_trimmed \
  --model_save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast \
  --use_train_test_split True \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 300 \
  --eval_freq 10 \
  --save_freq 10 \
  --obs_horizon 2 \
  --pred_horizon 12 \
  --action_horizon 4 \
  --num_diffusion_iters 100 \
  --predict_pos_delta True \
  --image_output_size 64 \
  --color_jitter True \
  --state_noise 0.005 \
  --use_memmap_cache True \
  --load_img False \
  --gpu 0  \
  --use_wandb True \
  --wandb_entity_name shiyi_gu_seu-org \
  --wandb_project_name tele-gsy \
  --wandb_exp_name rubiks_cube_trimmed_dp_img_pos_delta_fast
  

解释：

python learning/dp/pipeline.py \
  --data_path /home/shiyigu/Documents/project/tele-gsy/data_split/rubiks_cube_trimmed 数据集路径前缀，配合 --use_train_test_split True，实际读取 _train 和 _test 两个目录\
  --model_save_path /home/shiyigu/Documents/project/tele-gsy/data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast 模型保存目录。代码会在这个目录下再创建一个带时间戳和参数信息的子目录，保存 ckpt、stats、args log\
  --use_train_test_split True 使用 train/test 拆分数据\
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 这里可以调大点 \
  --num_workers 4 # DataLoader 用 4 个 worker 读数据\
  --epochs 300 \
  --eval_freq 10 每 10 个 epoch 做一次 eval\
  --save_freq 10 每 10 个 epoch 保存一次模型\
  --obs_horizon 2 \
  --pred_horizon 12 \
  --action_horizon 4 \
  --num_diffusion_iters 100 扩散模型去噪步数是 100\
  --predict_pos_delta True 预测 joint position 的 delta，也就是动作更偏向“相对变化量”，不是绝对目标位置。这通常比直接预测绝对关节位置更稳。\
  --image_output_size 64 每个相机输出 64 维特征\
  --color_jitter True 对 RGB 图像做亮度扰动，增加视觉鲁棒性\
  --state_noise 0.005 给状态输入加小噪声，增强模型对关节状态误差的鲁棒性\
  --use_memmap_cache True 使用之前准备好的 memmap 图像缓存\
  --load_img False 不把所有图片一次性读进内存\
  --gpu 0 使用第 0 张 GPU

--model_save_path生成文件：

- ema_last.ckpt: last.ckpt 对应的 EMA 权重。EMA 是 Exponential Moving Average，训练时对模型参数做滑动平均，通常用于推理/测试更稳。
- last.ckpt: 当前训练网络 nets 在最后一个 epoch 的权重。训练过程中每个 epoch 都会覆盖保存一次，所以它永远代表“最后状态”。当前加载逻辑会先加载你传入的普通 ckpt 到 nets，然后自动查找同目录下同名加 ema_ 前缀的文件，所以跑测试的时候入参写这个就行。
- model_epoch_290.ckpt: 普通训练权重
- ema_model_epoch_290.ckpt: 对应的 EMA 权重
- stats.pkl: 数据归一化/反归一化用的统计量，比如 action、pos 等的 min/max 或统计信息。测试时必须和 checkpoint 放在同一个文件夹下，因为 Agent.load() 会自动读
- full_eval_summary.pkl: 训练结束后代码对 test trajectories 跑 eval_all() 存下来的整体评估结果，里面有每条测试轨迹的 action_mse、normalized_mse，以及平均值

# Testing
gpt写的：

单条轨迹测试：
python learning/dp/pipeline.py \
  --eval True \
  --load_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast/0428_205543_RIzG-camera=01-identity=False-repr=IP-oh=2-ah=4-ph=12-prefix=None-do=0.0-imgos=64-wd=1e-05-use_ddim=False-binarize_touch=False-posdelta/last.ckpt \
  --eval_path /users/k25070928/project/tele-gsy/data_split/rubiks_cube_trimmed_test/0424_173508 \
  --save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/eval_results/last_traj_eval \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --obs_horizon 2 \
  --pred_horizon 12 \
  --action_horizon 4 \
  --num_diffusion_iters 100 \
  --predict_pos_delta True \
  --image_output_size 64 \
  --gpu 0

用整个test set测试：

eval_dir.py 会从 checkpoint 旁边的 args_log.txt 读训练参数，通常比手动跑 pipeline.py --eval True 更不容易配错

python eval_dir.py \
  --ckpt_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast/0428_205543_RIzG-camera=01-identity=False-repr=IP-oh=2-ah=4-ph=12-prefix=None-do=0.0-imgos=64-wd=1e-05-use_ddim=False-binarize_touch=False-posdelta/last.ckpt \
  --eval_dir /users/k25070928/project/tele-gsy/data_split/rubiks_cube_trimmed_test \
  --save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/eval_results/eval_last.pkl

结果保存到  --save_path，打印每个 checkpoint 的 MSE，例如 MSE for ...

真机测试：

1 启动硬件节点

python launch_nodes.py \
  --robot ur \
  --robot-ip 10.40.101.10 \
  --cam-names first_view third_view

2 测试

python run_env.py \
  --agent dp \
  --dp-ckpt-path data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast/0428_205543_RIzG-camera=01-identity=False-repr=IP-oh=2-ah=4-ph=12-prefix=None-do=0.0-imgos=64-wd=1e-05-use_ddim=False-binarize_touch=False-posdelta/last.ckpt \
  --hz 15 \
  --safe \
  --save-data \
  --data-dir ./shared/data/bc_data/dp_rollouts \

如果 DP 是用 marker tracking overlay 版触觉视频训练的，例如
`shared/data/bc_data/put_bottle_upright_tactile_crop` 中的
`/videos/tactile_left_rgb` 和 `/videos/tactile_right_rgb` 已经被替换成带箭头版本，
测试时需要让实时 marker tracking 结果作为 policy 输入：

python run_env.py \
  --agent dp \
  --dp-ckpt-path <your_marker_tracking_dp_ckpt> \
  --hz 15 \
  --safe \
  --use-tactile \
  --enable-marker-tracking \
  --tactile-crop-config-dir sensor_configs/put_bottle_upright \
  --use-marker-tracking-overlay-for-policy

`--tactile-crop-config-dir` 会把实时触觉图像先按该任务的裁剪配置处理，
再生成 marker tracking 箭头图；因此弹出的
`tactile_left_marker_tracking` / `tactile_right_marker_tracking` 窗口显示的也是裁剪后的版本。
这些裁剪配置默认假设输入图像尺寸为 320x240；传入 `--tactile-crop-config-dir`
后，实时触觉相机也会先输出 320x240，和训练 H5 中用于选点的图像尺寸保持一致。
如果你的训练数据使用了其他触觉图像尺寸，可以用 `--tactile-crop-input-width`
和 `--tactile-crop-input-height` 调整。

程序启动后会先把机器人移动到 reset joints，然后提示：
- 按一下并松开键盘 r：移动到初始位置并开始执行 policy
- 双击 r：停止并保存当前 trajectory
- 三击 r：停止并删除当前 trajectory
- Ctrl+C：退出程序

问题：safe是什么?dp_rollouts要改

# pi0环境
官方仓库：https://github.com/Physical-Intelligence/openpi

在tele环境用pip install uv就可以装uv
安装命令在openpi目录下运行：GIT_LFS_SKIP_SMUDGE=1 uv sync
第二句改成：GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e .

uv装在tele环境里，用uv管理venv环境
.venv/bin/python

If your dataset uses different names, update learning/pi0_ur5e/configs/dataset_schema.yaml before conversion

## 数据转换
cd openpi
uv run python /scratch/grp/luo/shiyi/project/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py \
  --input-root /scratch/grp/luo/shiyi/project/tele-gsy/shared/data/bc_data/put_golf_ball \
  --output-root /scratch/grp/luo/shiyi/project/tele-gsy/outputs/put_golf_ball_lerobot_no_tactile \
  --config /scratch/grp/luo/shiyi/project/tele-gsy/learning/pi0_ur5e/configs/dataset_schema.yaml \
  --task-name put_golf_ball \
  --repo-id local/pi0_ur5e_put_golf_ball_no_tactile \
  --default-prompt "Put the golf ball into the small container on the tray" \
  --action-mode joint_position_gripper \
  --include-tactile false \
  --overwrite true

解释：
uv run python /scratch/grp/luo/shiyi/project/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py \
  --input-root /scratch/grp/luo/shiyi/project/tele-gsy/shared/data/bc_data/put_golf_ball \
  --output-root /scratch/grp/luo/shiyi/project/tele-gsy/outputs/put_golf_ball_lerobot_no_tactile \
  --config /scratch/grp/luo/shiyi/project/tele-gsy/learning/pi0_ur5e/configs/dataset_schema.yaml \
  --task-name put_golf_ball \
  --repo-id local/pi0_ur5e_put_golf_ball_no_tactile \
  --default-prompt "Put the golf ball into the small container on the tray" \
  --action-mode joint_position_gripper \
  --include-tactile false \
  --overwrite true

## 训练
训练 pi-0 不是直接手动进 openpi 跑一堆命令，而是用本仓库的 helper 脚本 learning/pi0_ur5e/scripts/train_pi0_base.sh 来帮你把准备工作和训练串起来（todo 每个任务需要修改脚本里的参数，至少prompt需要改）
- 把我们这个仓库里的 UR5e 自定义配置安装/写入到 OpenPI 项目里：机器人数据长什么样、state/action 维度是多少、用哪个 LeRobot dataset、checkpoint 存哪里。虽然名字叫 cup，但这里更像是一个通用 UR5e pi0 配置名，不一定只能训练 cup 任务
- links the LeRobot dataset into a local HF_LEROBOT_HOME，把实际存储数据的位置（/scratch/grp/luo/shiyi/project/tele-gsy/outputs/put_golf_ball_lerobot_no_tactile
）建立一个软链接到代码读取的位置
- reads observation.state.shape from meta/info.json，这是转换后的 LeRobot 数据集，读维度来改模型输入层维度
- 算归一化统计量：runs compute_norm_stats.py，这个步骤会生成 OpenPI 训练需要的 normalization assets。
- 正式开始训练：runs OpenPI training

如果用Pi0而不是pi0.5的话，要用下面的命令（使用 LoRA 微调）：
--model-family pi0 --pi05 false --lora true
LoRA 的意思是：不全量更新整个大模型参数，只训练一小部分低秩适配参数

--use-delta-actions true意思是：你转换出来的数据 action 仍然保存为绝对关节位置格式

### 无触觉训练
cd /scratch/grp/luo/shiyi/project/tele-gsy
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root /scratch/grp/luo/shiyi/project/tele-gsy/outputs/put_golf_ball_lerobot_no_tactile \
  --output-dir /scratch/grp/luo/shiyi/project/tele-gsy/outputs/pi0_put_golf_ball_no_tactile_lora \
  --openpi-root /scratch/grp/luo/shiyi/project/openpi \
  --repo-id local/pi0_ur5e_put_golf_ball_no_tactile \
  --exp-name put_golf_ball_pi0_base_no_tactile_lora \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile false \
  --default-prompt "put the golf ball into the small container on the tray"
  --dry-run true

## server运行命令
### 无触觉
cd /home/kun/Yongqiang/tele-gsy
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/kun/Yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir /home/kun/Yongqiang/tele-gsy/outputs/pi0_fold_Tshirt_no_tactile_lora/checkpoints/pi0_ur5e_cup/fold_Tshirt_pi0_base_no_tactile_lora/13000 \
  --dataset-root /home/kun/Yongqiang/tele-gsy/outputs/fold_Tshirt_lerobot_no_tactile \
  --model-family pi0 \
  --use-delta-actions true \
  --camera-padding-strategy zeros \
  --default-prompt "Fold the t-shirt in half" \
  --port 8000

### 有触觉
cd /home/kun/Yongqiang/tele-gsy
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/kun/Yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir /home/kun/Yongqiang/tele-gsy/outputs/pi0_fold_Tshirt_tactile_emb_lora/checkpoints/pi0_ur5e_cup/fold_Tshirt_pi0_base_tactile_emb_lora/29999 \
  --dataset-root /home/kun/Yongqiang/tele-gsy/outputs/fold_Tshirt_lerobot_tactile_emb \
  --model-family pi0 \
  --use-delta-actions true \
  --camera-padding-strategy zeros \
  --default-prompt "Fold the t-shirt in half" \
  --port 8000

### 真机测试 tactile embedding pi0
如果 pi0 是用 marker tracking overlay 版触觉图像训练的，`run_env.py`
这边也要实时生成同样的裁剪后带箭头触觉图，再送进 pi0 的 tactile embedding：

python run_env.py \
  --agent pi0 \
  --pi0-policy-host 127.0.0.1 \
  --pi0-policy-port 8000 \
  --pi0-prompt "put the bottle upright" \
  --pi0-include-tactile \
  --pi0-tactile-feature-mode image_embedding \
  --use-tactile \
  --enable-marker-tracking \
  --tactile-crop-config-dir sensor_configs/put_bottle_upright \
  --use-marker-tracking-overlay-for-policy \
  --hz 15 \
  --safe

这时 pi0 的 `tactile_left_rgb` / `tactile_right_rgb` 输入，以及
`tactile_left_marker_tracking` / `tactile_right_marker_tracking` 显示窗口，
都会使用任务裁剪后的 marker tracking overlay。
