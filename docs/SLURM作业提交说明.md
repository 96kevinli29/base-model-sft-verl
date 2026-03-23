# SLURM 作业提交说明（LXP / MeluXina 集群）

依据 [LXP 官方文档 - Handling jobs](https://docs.lxp.lu/first-steps/handling_jobs/) 整理，用于在 **MeluXina** 或同架构 LXP 集群上提交 GPU 训练作业。

---

## 要点

- **不要在登录节点跑训练**：长时间任务和 GPU 程序必须在计算节点上跑，通过 SLURM 提交。
- **两种用法**：
  - **交互式 (dev)**：`srun` 连到计算节点，现场敲命令调试。
  - **批处理 (batch)**：写一个脚本，用 `sbatch` 提交，作业在后台跑，输出写到日志文件。
- **必填 SBATCH 参数**：`--account`（项目账号，如 p201251）、`--partition`、`--qos`、`--time`、`--nodes`、`--cpus-per-task`。

---

## 分区与 QOS（MeluXina）

| 分区 (partition) | 节点         | 说明           |
|------------------|--------------|----------------|
| **gpu**         | mel[2001-2200] | GPU 节点，跑 verl 用这个 |
| cpu             | mel[0001-0573] | 纯 CPU         |
| largemem        | mel[4001-4020] | 大内存         |

| QOS      | 最长时间 | 说明           |
|----------|----------|----------------|
| **dev**  | 6 小时   | 交互开发，每用户 1 个 job |
| **test** | 30 分钟  | 测试/调试，每用户 1 个 job |
| **default** | 48 小时 | 常规生产作业   |
| short    | 6 小时   | 短作业         |
| long     | 144 小时 | 长作业，每用户 1 个 job |

---

## 提交方式一：批处理（推荐跑训练）

1. 在项目目录下创建作业脚本（见下方示例），例如 `run_verl_gpu.sh`。
2. 在登录节点执行：
   ```bash
   cd /project/home/p201251/hyl_ppo
   sbatch run_verl_gpu.sh
   ```
3. 会返回作业 ID（如 `358492`），用 `squeue -u $USER` 查看排队/运行，用 `sacct -j 358492` 查看状态。

---

## 提交方式二：交互式（调试用）

申请 1 个 GPU 节点、进入 dev QOS 交互会话（最多 6 小时）：

```bash
srun --account=p201251 --partition=gpu --qos=dev --time=01:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 \
  --pty bash -l
```

进入后先激活环境再跑命令：

```bash
cd /project/home/p201251/hyl_ppo
source activate_verl.sh
python -c "import torch; print(torch.cuda.is_available())"
# 然后跑你的训练或调试命令
```

---

## 如果链接打不开

- 文档地址：<https://docs.lxp.lu/first-steps/handling_jobs/>
- 若学校/公司网络无法访问，可以：
  1. 用校园 VPN 或单位代理再试；
  2. 按本文档和下面的示例脚本操作，核心用法已包含在此；
  3. 在集群上查本地文档：`man sbatch`、`man srun`，或问管理员是否有内部 SLURM 说明页面。

---

## GPU 作业脚本示例

下面是一个**最小可用的 GPU 批处理脚本**，只做环境验证；真正跑 verl 时把最后的 `python -c ...` 换成你的训练命令（如 `python -m verl.trainer.run_ppo ...`）。

```bash
#!/bin/bash -l
#SBATCH --job-name=verl-gpu
#SBATCH --account=p201251
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/job.%j.out
#SBATCH --error=logs/job.%j.err

# 进入项目目录并激活 conda 环境
cd /project/home/p201251/hyl_ppo
source activate_verl.sh

# 示例：仅验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 正式训练时在这里写你的命令，例如：
# python -m verl.trainer.run_ppo ...
```

**注意**：  
- `--account=p201251` 请改成你的实际项目账号（和 `$HOME` 路径里的 p201251 一致即可）。  
- 提交前建议先建日志目录：`mkdir -p logs`，否则 `--output`/`--error` 可能报错。

---

## 常用 SLURM 命令

| 命令 | 作用 |
|------|------|
| `sbatch 脚本.sh` | 提交批处理作业 |
| `squeue -u $USER` | 看自己的作业队列 |
| `scancel 作业ID` | 取消作业 |
| `sacct -j 作业ID` | 查看作业状态/资源使用 |
| `srun ... --pty bash -l` | 交互式申请节点 |

更多选项见官方 [SLURM 快速参考](https://slurm.schedmd.com/quickstart.html)。
