# Round 5：强 RGB 上界与检测器换代

本轮停止 F003/F004 融合训练。两条正式主线均只用 RGB 和同一份固定 `1600/400`、`labels_clean` 数据合同，以便先回答“检测器上界是否足够高”。GPU-A 跑 YOLO11x/1280；GPU-B 跑 DEIMv2-S。IR 只做表示审计，通过门禁后才允许进入后续融合。

下列命令中的 `ROUND5_SHA` 必须替换为本次提交后报告的完整 40 位 SHA。两台机器各自 checkout 后都必须是 detached HEAD 且工作区为空。

## 0. 公共变量与 checkout

GPU-A：

```bash
export ROUND5_SHA='<40-char-commit-from-Codex>'
export AIC_REPO=/root/data1/AIC2026/AIC2026-Multimodal-Detection
cd "$AIC_REPO"
git fetch origin
git checkout --detach "$ROUND5_SHA"
test "$(git rev-parse HEAD)" = "$ROUND5_SHA"
test -z "$(git status --porcelain)"
```

GPU-B：

```bash
export ROUND5_SHA='<40-char-commit-from-Codex>'
export AIC_REPO=/root/data1/AIC2026/AIC2026-Multimodal-Detection-B
cd "$AIC_REPO"
git fetch origin
git checkout --detach "$ROUND5_SHA"
test "$(git rev-parse HEAD)" = "$ROUND5_SHA"
test -z "$(git status --porcelain)"
```

## 1. GPU-A：YOLO11x/1280

### 1.1 权重与数据视图

```bash
cd "$AIC_REPO"
source /root/data1/AIC2026/aic2026_env/bin/activate
mkdir -p weights
curl -L --fail --retry 5 \
  'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt' \
  -o weights/yolo11x.pt
echo '7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24  weights/yolo11x.pt' | sha256sum -c -
python scripts/train/verify_round5_assets.py --only yolo
python scripts/train/prepare_rgb_yolo.py --link-mode auto --force
```

`prepare_rgb_yolo.py` 必须报告 `train=1600`、`val=400`，并生成 `data/processed/rgb_yolo_clean/manifest.json`。训练入口还会再次核验标签聚合 SHA 和权重 SHA。

### 1.2 配置检查与全分辨率 smoke

```bash
python scripts/train/train_rgb.py \
  --config configs/experiments/RGB_R5_X1280.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/train/train_rgb.py \
  --config configs/experiments/RGB_R5_X1280.yaml \
  --smoke-full 2>&1 | tee /root/data1/AIC2026/RGB_R5_X1280_SMOKE_FULL.log
```

smoke 必须满足：CUDA 可用；实际 `imgsz=1280`、`batch=2`；完成 forward、backward 和一次验证；无 OOM、NaN、数据合同或 SHA 错误。若 OOM，只允许把正式配置的 batch 从 2 降到 1并形成新 commit；不得在服务器手改后直接正式训练。

### 1.3 正式训练

删除 smoke 产生的同名临时目录不影响正式目录；再次确认项目仓库 clean，然后执行：

```bash
cd "$AIC_REPO"
test "$(git rev-parse HEAD)" = "$ROUND5_SHA"
test -z "$(git status --porcelain)"
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_rgb.py \
  --config configs/experiments/RGB_R5_X1280.yaml \
  --train --expected-sha "$ROUND5_SHA" \
  2>&1 | tee /root/data1/AIC2026/RGB_R5_X1280_FORMAL.log
```

训练观察门槛：第 60 epoch 前 best `mAP50-95 < 0.485` 时先继续到 90；第 90 epoch 仍 `< 0.490`，停止并保留 best.pt、results.csv、args.yaml 和日志。若达到或超过 0.50，继续至 early stop/180 epoch。这个门槛只节省算力，不改变正式验证协议。

## 2. GPU-B：DEIMv2-S / DINOv3 蒸馏骨干

### 2.1 独立 Python 3.11 环境

DEIMv2 不复用 Ultralytics 的 Python 3.8 环境。官方固定源码的 requirements 使用 PyTorch 2.5.1；这里显式安装 CUDA 11.8 wheel：

```bash
export DEIM_ENV=/root/data1/AIC2026/envs/deimv2_py311
conda create -p "$DEIM_ENV" python=3.11 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DEIM_ENV"
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

固定源码并安装除 torch/torchvision 外的上游依赖：

```bash
export DEIM_ROOT=/root/data1/AIC2026/source_packages/DEIMv2
if [ ! -d "$DEIM_ROOT/.git" ]; then
  git clone https://github.com/Intellindust-AI-Lab/DEIMv2.git "$DEIM_ROOT"
fi
git -C "$DEIM_ROOT" fetch origin
git -C "$DEIM_ROOT" checkout --detach 1d2ca42171570c713e78fc6a766ec5104b7f4724
test "$(git -C "$DEIM_ROOT" rev-parse HEAD)" = '1d2ca42171570c713e78fc6a766ec5104b7f4724'
test -z "$(git -C "$DEIM_ROOT" status --porcelain)"
grep -Ev '^(torch|torchvision)==' "$DEIM_ROOT/requirements.txt" > /tmp/deimv2_requirements_no_torch.txt
python -m pip install -r /tmp/deimv2_requirements_no_torch.txt
python -m pip freeze > /root/data1/AIC2026/deimv2_py311_freeze.txt
python - <<'PY'
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
assert torch.cuda.is_available()
PY
```

若服务器没有 `conda`，用已安装的 Python 3.11 创建同一路径：`python3.11 -m venv "$DEIM_ENV"`，然后 `source "$DEIM_ENV/bin/activate"`，其余命令不变。

### 2.2 官方权重下载与强校验

```bash
cd "$AIC_REPO"
mkdir -p weights/deimv2
curl -L --fail --retry 5 \
  'https://drive.usercontent.google.com/download?id=1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL&export=download&confirm=t' \
  -o weights/deimv2/deimv2_dinov3_s_coco.pth
curl -L --fail --retry 5 \
  'https://drive.usercontent.google.com/download?id=1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs&export=download&confirm=t' \
  -o weights/deimv2/vitt_distill.pt
echo '9491ab33b68ecfc0e34043abb3009599ab1e892fb953a1faad12ef4fca5a35c4  weights/deimv2/deimv2_dinov3_s_coco.pth' | sha256sum -c -
echo '2053b865f4e2673fba3f95f7e7e54ad5ee18143885e3ad27eaabb5b3b9919738  weights/deimv2/vitt_distill.pt' | sha256sum -c -
python scripts/train/verify_round5_assets.py --only deim --deim-root "$DEIM_ROOT"
```

### 2.3 COCO 导出与运行时配置

```bash
cd "$AIC_REPO"
python scripts/data/prepare_deimv2_coco.py --link-mode auto --force
mkdir -p data/processed/deimv2_runtime
python scripts/train/prepare_deimv2_runtime.py \
  --deim-root "$DEIM_ROOT" --profile smoke \
  --output data/processed/deimv2_runtime/deimv2_s_smoke.yml
python scripts/train/prepare_deimv2_runtime.py \
  --deim-root "$DEIM_ROOT" --profile formal \
  --output data/processed/deimv2_runtime/deimv2_s_formal.yml
```

导出必须生成 1600/400 张图、12 类、类别 ID `0..11`，`manifest.json` 中的 labels 与原始 visible 图片 aggregate SHA 必须匹配。配置生成器会重新读取导出的图片和 annotation 文件，拒绝错误源码 commit、脏源码 checkout、权重哈希或数据合同。

跨平台统一参考值为：train `12153` 个框、annotation SHA256 `7a0c369ee0c4ca0a15858436904fadcbf6fb105741c43611498ed9c2d5f88536`；val `3041` 个框、annotation SHA256 `d33c458bc9d09cd40f609c12e8548a96757cedcb88609f53d503831a32436a6a`。原始 visible 图片 aggregate SHA256 分别是 train `a30d51404d21017766434fbfc03b07e89101d4121a61ec3239b1b70de51e3626`、val `92b505f0391ba3f084b9c677ba1907f1a2ed26af9cec6c1e85f0189023718a0a`。服务器重导出必须全部匹配，否则停止 smoke。旧 Windows annotation SHA 只因文本模式写入 `CRLF` 而不同，见 `docs/DEIMV2_COCO_SHA_DIAGNOSTIC.md`。

### 2.4 smoke 与正式训练

```bash
export DEIM_CKPT="$AIC_REPO/weights/deimv2/deimv2_dinov3_s_coco.pth"
export DEIM_SMOKE_CFG="$AIC_REPO/data/processed/deimv2_runtime/deimv2_s_smoke.yml"
export DEIM_FORMAL_CFG="$AIC_REPO/data/processed/deimv2_runtime/deimv2_s_formal.yml"

cd "$DEIM_ROOT"
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c "$DEIM_SMOKE_CFG" -t "$DEIM_CKPT" --use-amp --seed 2026 \
  --output-dir /root/data1/AIC2026/runs/DEIMV2_S_SMOKE \
  2>&1 | tee /root/data1/AIC2026/DEIMV2_S_SMOKE.log
```

smoke 必须完成 2 epoch、训练/验证均有有限 loss、COCO evaluator 能输出 12 项 bbox 指标、没有类别越界、OOM 或 NaN。随后再次执行项目和 DEIMv2 两边的 SHA/clean 检查，再正式训练：

```bash
test "$(git -C "$AIC_REPO" rev-parse HEAD)" = "$ROUND5_SHA"
test -z "$(git -C "$AIC_REPO" status --porcelain)"
test "$(git -C "$DEIM_ROOT" rev-parse HEAD)" = '1d2ca42171570c713e78fc6a766ec5104b7f4724'
test -z "$(git -C "$DEIM_ROOT" status --porcelain)"
cd "$DEIM_ROOT"
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c "$DEIM_FORMAL_CFG" -t "$DEIM_CKPT" --use-amp --seed 2026 \
  --output-dir /root/data1/AIC2026/runs/DEIMV2_S_RGB_FORMAL \
  2>&1 | tee /root/data1/AIC2026/DEIMV2_S_RGB_FORMAL.log
```

DEIMv2 的 `log.txt` 中 `test_coco_eval_bbox[0]` 是 mAP50-95。第 36 epoch 前只观察；第 48 epoch best 仍 `< 0.47` 时停止，说明换检测器未形成有效上界。达到 `>=0.50` 则完成 72 epoch 两阶段日程，并保存 `best_stg1.pth`、`best_stg2.pth`、`last.pth`、`log.txt`、两份运行时 YAML 和 freeze 文件。

## 3. IR heat residual 门禁

在任一项目 checkout 上、启动正式训练前运行；这是 CPU 只读审计，不生成训练数据：

```bash
cd "$AIC_REPO"
source /root/data1/AIC2026/aic2026_env/bin/activate
python scripts/analysis/audit_ir_heat_residual.py \
  --expected-sha "$ROUND5_SHA" \
  --output-dir outputs/analysis/IR_HEAT_AUDIT_R5
echo "IR audit exit code=$? (0=PASS, 3=REJECT)"
```

审计只在可信局部边缘配准样本上评估 `max(IR_aligned - RGB_gray, 0)`，目标类固定为 person 和 animal。以下五项必须全部通过：

1. 可信配准图像比例 `>= 20%`；
2. 可评估热目标框数 `>= 50`；
3. heat residual SNR 为正的目标比例 `>= 60%`；
4. heat residual 的中位 SNR `>= 0.75`；
5. 相对原始 IR 的逐框 SNR 增益中位数 `>= 0.10`。

`summary.json` 的 `decision=PASS` 且退出码为 0 才允许设计下一轮 heat-residual 融合。任一项失败即 `REJECT`（退出码 3），不启动该表示的融合训练；这时保留 RGB 强模型，转向检测器结构、输入分辨率和单模型推理优化。

本地开发态已用完整 val400 做过一次不带正式 checkout 证明的预检：可信配准 `101/400`、可评估热目标框 `287`、heat 正 SNR 比例 `65.16%`、heat 中位 SNR `0.4747`、相对 raw IR 中位增益 `0.1686`。它只因绝对信号未达到 `0.75` 而得到 `REJECT`。服务器仍须按上面的正式命令复核，但除非正式结果跨过同一固定阈值，本轮不训练 IR heat fusion。

## 4. 回传 artifact

GPU-A 回传：完整日志、`results.csv`、`args.yaml`、`results.png`、best.pt、last.pt 和 best.pt 独立 reval 指标。GPU-B 回传：完整日志、`log.txt`、运行时 YAML、环境 freeze、best_stg1/stg2、last 和最佳 epoch。IR 回传 `summary.json` 与 `targets.csv`。三个目录分别打包，禁止混放或只抄最终数字。
