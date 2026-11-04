# Mac MuJoCo + DeepSeek agent

根目录是独立的 uv 项目，使用 Python 3.12、MuJoCo、NumPy、OpenCV、ZMQ、dotenv 和 HTTPX。
它不安装 LeRobot、Torch、CUDA、Pinocchio 或 CasADi。原有 LeRobot 子项目继续使用自己的环境。

## 启动

```bash
uv sync --locked
./run_hei_robot_policy_sim_on_mac.sh
```

在已登录的 Mac 桌面会话中运行，会显示机器人、桌子、红绿蓝方块和香蕉。
窗口右侧默认实时显示 `FRONT`（前置）、`LEFT WRIST`（左腕）、`RIGHT WRIST`（右腕）三路画面。
预览复用发布给 agent 的同一批相机 RGB 帧，不增加第二次场景渲染，窗口缩放时保持画面比例。
为留出画面空间，MuJoCo 自带的左右控制栏默认收起，可用 `Tab` / `Shift+Tab` 切换。
不需要预览时运行 `./run_hei_robot_policy_sim_on_mac.sh --no-camera-preview`。
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
成功要求仿真反馈中 `cube_blue.held_by` 非空、物体中心距夹爪 TCP 不超过 1.5 cm，
且蓝块比起始位置升高至少 10 cm。

## DeepSeek

将 `.env.example` 复制为 `.env`（已有 `.env` 时保留它），填写：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_TIMEOUT_SEC=60
DEEPSEEK_TRUST_ENV=false
DEEPSEEK_MAX_RETRIES=2
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

API 默认直连 HTTPS，`DEEPSEEK_TRUST_ENV=false` 避免 HTTPX 自动读取 Mac 系统代理；
即使终端没有 `HTTPS_PROXY`，系统代理仍可能影响 HTTPX。需要代理时将该值改为 `true`，
恢复 HTTPX 对环境／系统代理和证书环境变量的读取。TLS 证书验证始终开启。
连接重置、超时和部分临时 HTTP 错误会有限重试，默认最多重试 2 次（总共 3 次请求）；
认证、余额或参数错误直接退出。重试只重发 API 请求，不重放已执行的动作工具；
每次尝试前刷新观测并检查仿真会话，等待期间由原 I/O 线程继续保持目标。

遇到 `[Errno 54] Connection reset by peer`，可以先只检查 API，不启动也不连接仿真：

```bash
./run_hei_robot_agent.sh --check-api
```

该命令会发送一次最小文本请求（失败时按配置重试），成功打印 `HTTP 200: OK`。
它验证 API 连通性和密钥；图像输入与完整任务仍需通过实际 agent 运行验证。

可选参数：`--text-only`（只发送结构化状态）、`--max-steps 12`（最多 30 轮）。
工具调用、返回结果和 API usage 保存在 `outputs/sim-agent/*.jsonl`，不记录密钥或图像数据。

此版本使用**明确标注的仿真真值**：物体名称、世界坐标、TCP 坐标、夹持状态。
它不是仅凭图像估计三维位置的 VLA，也不加载 `060000` 权重。
抓取仍使用运动学辅助附着：先移动到物体中心（`clearance_m=0`），再闭爪；
只有物体中心距 TCP 不超过 1.5 cm 才允许附着，保留这段小位姿误差。
旧版 11 cm 抓取半径会导致隔空抓取，现已收紧；更新后需重启仿真和 agent。
这仍是几何距离判定，未验证夹指接触、摩擦、夹持力或夹爪姿态，不能作为物理抓取成功的证明。
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
