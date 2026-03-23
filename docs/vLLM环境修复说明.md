# vLLM 与 PyTorch 不兼容时的修复说明

## 现象

运行六数据集 benchmark 或其它用到 vLLM 的脚本时报错：

```text
ImportError: .../vllm/_C.abi3.so: undefined symbol: _ZN3c106ivalue14ConstantString6createE...
```

## 原因

vLLM 的 C++ 扩展（`_C.abi3.so`）是按**某一版本 PyTorch** 编译的。当前环境里的 PyTorch 与当时编译用的版本不一致（或 ABI 不同），就会出现上述 undefined symbol。

常见情况：

- 先装了 vLLM，后来升级/重装了 PyTorch
- 用 pip 装了 vLLM 的预编译 wheel，该 wheel 针对的 PyTorch 与你当前版本不同
- 环境里存在多份 PyTorch（如 conda + pip 混用）

你当前环境：**torch 2.6.0+cu124**、**vLLM 0.8.5.post1**（见 `环境配置说明-学校与云端.md`）。

---

## 修复方式（任选其一）

### 方式一：强制重装 vLLM（优先试）

在 **verl** 环境中执行，让 pip 重新拉取与当前环境匹配的 wheel：

```bash
source activate_verl.sh
pip uninstall vllm -y
pip install vllm==0.8.5.post1 --no-cache-dir
```

若仍报同样错误，再用方式二。

### 方式二：从源码编译 vLLM（与当前 PyTorch 一致）

在**带 GPU 和 CUDA 的计算节点**上做（避免登录节点无驱动/库不全）：

```bash
source activate_verl.sh
pip uninstall vllm -y
# 从源码安装，会按当前 PyTorch 编译
pip install vllm --no-binary vllm --no-cache-dir
```

要求：

- 计算节点上有 CUDA、编译链（gcc、CMake ≥ 3.26.1）
- 耗时可能 10–30 分钟，视节点性能而定

若在登录节点没有 GPU，可写一个短 SLURM 作业，在计算节点上执行上述两条 `pip` 命令（或交互式 `salloc` 进去再执行）。

### 方式三：临时不用 vLLM，用 HF 后端跑通

不修 vLLM 也能跑六数据集 benchmark，改用 transformers 后端：

```bash
BACKEND=hf bash scripts/run_six_sets_test.sh
```

或在 `run_six_sets_test.sh` 里给 `run_benchmark.py` 增加参数：`--backend hf`。速度会慢一些，但能出结果。

---

## 验证 vLLM 是否正常

在**计算节点**（有 GPU 的环境）上：

```bash
source activate_verl.sh
python -c "from vllm import LLM; print('vLLM OK')"
```

无报错即表示当前环境里 vLLM 可用。

---

## 参考

- [vLLM Issue #5501](https://github.com/vllm-project/vllm/issues/5501)（同类 symbol 错误）
- 项目内 `verl/docs/README_vllm0.8.md`：verl 推荐的 vLLM 0.8 安装顺序（先 PyTorch，再 vLLM）
