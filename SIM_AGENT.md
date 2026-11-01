# Mac MuJoCo + DeepSeek agent

根目录是独立的 uv 项目，使用 Python 3.12、MuJoCo、NumPy、OpenCV、ZMQ、dotenv 和 HTTPX。
它不安装 LeRobot、Torch、CUDA、Pinocchio 或 CasADi。原有 LeRobot 子项目继续使用自己的环境。

## 启动

```bash
uv sync --locked
./run_hei_robot_policy_sim_on_mac.sh
```

在已登录的 Mac 桌面会话中运行，会显示机器人、桌子、红绿蓝方块和香蕉。
脚本使用 MuJoCo 官方 `mjpython` 启动器以及 CGL 离屏相机渲染。
启动器会为 uv 管理的 Python 补上 `libpython` 动态库搜索路径，避免 `.venv` 下的 `dlopen` 错误。
无需设置 `DISPLAY`、Xvfb 或 GLX；脚本会清除这些 Linux 专用环境变量。
窗口中按 `R` 重置，关闭窗口退出。重置后需要重新启动 agent。

另开一个终端，先验证不依赖 API 的固定动作计划：

```bash
./run_hei_robot_agent.sh --demo
```

这会张开左夹爪、移动到蓝块上方、靠近蓝块、闭合夹爪，再抬起 15 cm。
它明确是固定计划，不是语言模型控制。完成后保持抓取姿态，不自动松爪或回零。
成功要求仿真反馈中 `cube_blue.held_by` 非空，且蓝块比起始位置升高至少 10 cm。

## DeepSeek

将 `.env.example` 复制为 `.env`（已有 `.env` 时保留它），填写：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT_SEC=60
HEI_AGENT_SEND_IMAGES=true
```

`.env` 已被 Git 忽略。不要将密钥写入命令、文档或日志。
先在窗口中按 `R` 复位，再启动：

```bash
./run_hei_robot_agent.sh
```

使用 `--task` 传入中文或英文任务，例如：

```bash
./run_hei_robot_agent.sh --task "抓起红色方块，抬高至少 10 厘米并保持夹住"
./run_hei_robot_agent.sh --task "抓起绿色方块，抬高至少 15 厘米并保持夹住"
./run_hei_robot_agent.sh --task "Pick up the blue cube and hold it at least 10 cm higher."
```

每条命令独立运行；切换任务前先结束上一个 agent，在仿真窗口中按 `R` 复位。
省略 `--task` 时默认为抓起蓝块。`--demo` 保留为固定蓝块演示，不接受 `--task`。

任务原文作为用户消息发送给 DeepSeek，由模型结合场景选择物体、手臂和动作顺序，
代码不按颜色关键词匹配任务。模型先调用 `select_target`，绑定目标名称及最低抬升高度；
该目标在本次运行中锁定。`done` 检查这个物体的夹持状态与实际抬升量，
抓起其他方块或降低完成标准不会被判为成功。任务原文和工具调用一并记录，可核对模型是否选对目标。

当前支持**单个物体的抓起、抬升并保持**（最低抬升要求 5–25 cm）。
语言理解和目标选择由模型完成，并不保证任意描述都能正确识别；歧义或不支持的任务应返回
`cannot_complete`。放置、堆叠、多物体连续操作尚未提供相应工具和成功判定。

默认发送前置、左腕、右腕三路相机 JPEG，以及结构化仿真状态。
DeepSeek 通过 `select_target`、`observe`、`set_gripper`、`move_to_object`、`lift`、`done` 工具选择动作。
本地 MuJoCo 雅可比 IK 将目标转换为关节角和升降高度，工具校验范围并等待反馈收敛。
独立 I/O 线程以 10 Hz 发送目标，API 等待不会中断仿真命令心跳。
网络断流、仿真重启或重置会中止动作；API 错误会退出，不会偷偷切换成固定演示。

可选参数：`--text-only`（只发送结构化状态）、`--max-steps 12`（最多 30 轮）。
工具调用、返回结果和 API usage 保存在 `outputs/sim-agent/*.jsonl`，不记录密钥或图像数据。

此版本使用**明确标注的仿真真值**：物体名称、世界坐标、TCP 坐标、夹持状态。
它不是仅凭图像估计三维位置的 VLA，也不加载 `060000` 权重。
抓取沿用项目原有的运动学辅助吸附：夹爪从张开转为闭合且 TCP 靠近物体时附着。
这能验证 agent 工具闭环，不等同于真实接触力学或实机抓取成功率。
agent 只连接本机仿真端口 `6565/6566`，并检查仿真类型，不提供真实机械臂控制入口。

## Linux 仿真

在根目录执行原有显示环境设置即可：

```bash
DISPLAY=:99 MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx LIBGL_ALWAYS_SOFTWARE=1 \
  env -u LD_LIBRARY_PATH ./run_hei_robot_policy_sim.sh \
  --control-source policy --publish-fps 10 --verbose
```

Xvfb/VNC 属于系统显示环境，不由 pip 安装。根目录启动脚本使用 uv。
原 `VR_mujoco_ik/run_hei_robot_policy_sim.sh` 在 uv 可用时转到根目录启动器；
需要沿用旧 Conda 安装时，设置 `HEI_SIM_RUNTIME=conda`。
其他 VR/键盘 IK 入口仍使用原 Pinocchio/CasADi 环境。

## 检查

```bash
uv run --locked python -m unittest discover -s tests_sim -v
uv tree --locked
```

测试使用真实 MuJoCo 模型与 action adapter，检查红绿蓝方块的 IK 可达性、张开/闭合边沿、抬升和释放，
以及动态目标、抬升门槛和错误目标不能通过成功判定。
不需要 DeepSeek key，也不需要打开窗口。

已在本机 Mac 验证：最小 uv 环境、原生可视窗口、三路 JPEG 发布、固定计划及真实
`deepseek-flash` 图像与工具调用。真实 API 测试共 6 轮，蓝块由左夹爪持有，
高度从 0.82 m 升至约 0.97 m，`done.success=true`。
加入动态任务后，用“不要抓蓝色方块，抓起红色方块并抬高至少 10 厘米”进行真实 API 测试：
模型选择 `cube_red` 与右臂，首次闭合未抓住后根据反馈降低位置重试，9 轮后抬升约 15 cm，
`done.success=true`。绿色方块已通过本地模型抓取测试，尚未单独做真实 API 测试。
Linux 入口已做脚本检查；本次没有在 Linux 上运行显示环境。
