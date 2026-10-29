# 固定图像 Mask：训练与部署

白色保留、黑色置零；训练与部署都在原图 resize 前应用固定 mask，仅处理指定相机。默认相机为 `observation.images.cam_high_rgb`，腕部相机不变。PNG 或逐行区间 JSON 均可加载，原图尺寸必须与 mask 一致。

## 已保存的部署资源

| 目录 | 最终文件 | 来源与用途 |
| --- | --- | --- |
| `data-v1` | `union_mask.png` | 历史 mask，保留用于旧 checkpoint |
| `data-1023-1027` | `mask.png` / `mask_ranges.json` | 五个数据集各抽 20 帧后取数据集交集；保留约 62.43%，可能裁掉运动目标 |
| `data-1023-1028-optimized-v1` | `optimized_mask.png` / `optimized_ranges.json` | 根据 225 个抽样帧扩展保留区域，遮挡约 1.23% |
| `data-1023-1028-table-conveyor-v1` | `table_conveyor_mask.png` / `table_conveyor_ranges.json` | 桌面与输送带固定区域，遮挡约 34.35%；区域外机械臂和软包也会被遮挡 |

后两套使用 1023、1024、1026、1027、1028 数据，各 45 帧，具体方法与限制见各目录 `report.json`。它们不是全量逐帧分割结果；报告中的本机 analysis 路径是来源记录，不是部署依赖。最终 PNG、JSON 和报告一起保留，生成中间产物不提交。旧抽样生成器 `build_mask.py` 当前缺失，不能用旧命令重新生成历史结果。

## 训练与部署开关

训练时在原有指令中添加：

```bash
--data.image_mask_enabled true \
--data.image_mask_path mask/data-1023-1028-table-conveyor-v1/table_conveyor_mask.png \
--data.image_mask_camera observation.images.cam_high_rgb
```

不使用图像 mask 时指定 `--data.image_mask_enabled false`。图像 mask 与 `--data.right_arm_loss_only` 是独立选项；当前 `package_multi_posttrain.yaml` 默认只监督右臂和右夹爪，双臂监督需显式设为 false。

部署默认继承 checkpoint 的 `lingbotvla_cli.yaml`；也可显式覆盖：

```bash
python -m deploy.lingbot_vla_policy --model_path /path/to/hf_ckpt \
  --norm_path /path/to/training_norm.json \
  --image-mask-enabled \
  --image-mask-path mask/data-1023-1028-table-conveyor-v1/table_conveyor_mask.png

python -m deploy.lingbot_vla_policy --model_path /path/to/hf_ckpt \
  --norm_path /path/to/training_norm.json \
  --no-image-mask-enabled
```

相对 mask 路径基于仓库根目录。checkpoint 不打包 mask 文件；部署要携带实际训练版本并覆盖失效的绝对路径。对同一 checkpoint 切换 mask 可测试输入敏感性；对比两种训练方案则需分别使用对应 checkpoint。只监督右臂的模型仍输出完整动作，执行端应忽略左臂和左夹爪预测。

## 可选轮廓整理

```bash
.venv/bin/python mask/regularize_mask.py \
  --input mask/data-1023-1027/mask.png \
  --output /path/to/new_mask.png \
  --close-size 35 --epsilon 3 --min-area 300
```

输出 PNG 和同名 `_ranges.json`。此操作改变保留区域，应检查效果；默认拒绝覆盖已有结果。

## 两张 H200：全量逐帧计算

使用 `build_mask_all_frames.py`，与旧抽样脚本独立。每帧先由本地 Qwen3-VL-4B 定位机械臂、软包、绿色桌面和滚轮带，再用 SAM 分割。每帧重新定位，不复用固定提示点；所有帧、所有数据集取并集，不再取交集。视频按帧数分配到两张 GPU，每卡各加载一套模型，流式解码，不保存抽帧或中间分割图。

在 `lingbot-vla` 根目录执行。启动脚本仅选择已有兼容环境，不安装、升级或下载任何依赖；自动尝试 `MASK_PYTHON`、同级 `lingbot-vla-v2/.venv`、当前 `.venv` 和 `.venv-mask-full`，不兼容的环境会跳过：

```bash
bash mask/run_all_frames.sh check

# 先试跑：每数据集首个视频的前 8 帧，共 40 帧。
bash mask/run_all_frames.sh smoke

# 检查 smoke 目录中的 masked_preview.jpg 后，全量运行。
bash mask/run_all_frames.sh all
```

默认目录结构：

```text
workspace/
  DATA/Dmetdata/1023-data-v2-uppick/ ... 1027-data-v3.4-uppick-side/
  model/Qwen3-VL-4B-Instruct/
  model/sam-vit-base/
  lingbot-vla/
  lingbot-vla-v2/.venv/  # 已有兼容环境（PyTorch 2.8.0、Transformers 4.57.3）
```

需在 GPU 主机上准备上述数据、完整本地模型及本仓库脚本；运行时不联网下载模型。GPU 节点须能访问所选已有环境，脚本不会把 Transformers 4.51.3 改造成支持 Qwen3-VL 的版本；如果所有已有环境都不兼容，会报错退出。启动时会清除旧 `PYTHONPATH` 覆盖并禁用用户级 site-packages，避免混入之前建议的 `.mask-deps`，仅影响脚本子进程。若目录不同，可设置 `MASK_DATA_ROOT`、`MASK_GROUNDER`、`MASK_SAM`；`MASK_PYTHON` 可指定现有兼容环境的 Python。默认 `CUDA_VISIBLE_DEVICES=0,1`、`MASK_GPUS=2`、`MASK_BATCH_SIZE=4`，显存不足时降低 batch size。

输出目录分别为 `mask/data-1023-1027-allframes-smoke/` 和 `mask/data-1023-1027-allframes-all/`，每个只包含：

- `mask.png`：原始分辨率二值图，白色保留、黑色置零，可直接用于训练和部署。
- `masked_preview.jpg`：一张对比拼图，包含五个数据集各一帧的原图和应用最终 mask 后的效果。

默认拒绝覆盖已有目录。重跑可用 `MASK_OUTPUT=/path/to/new-directory bash mask/run_all_frames.sh all`。旧 mask 不被覆盖。不保存检查点，任务中断需要重算；终端持续报告处理帧数、速度和预计剩余时间。JSON 定位结果格式错误会重试一次，仍失败或视频解码失败则终止，不静默跳帧。完成时核对实际处理帧数等于索引总数；当前五套数据为 561,499 帧。

训练配置：

```yaml
image_mask_enabled: true
image_mask_path: mask/data-1023-1027-allframes-all/mask.png
image_mask_camera: observation.images.cam_high_rgb
```

部署同样使用该 PNG，无需 JSON 或在线分割。逐帧并集消除了抽样间隙和跨数据集交集造成的删减，但自动定位、分割仍可能漏检或误检；并集也会累计误检，无法保证自动产生精确的完整运动边界，因此先检查 smoke 效果再跑全量。当前环境仅完成了索引、完整解码、分片、坐标转换与输出的 CPU 测试，双卡模型推理需在 H200 环境试跑验证。

模型接口参考：[Qwen3-VL 官方 Transformers 文档](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/qwen3_vl)。
