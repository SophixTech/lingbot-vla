# data-v1 头部相机 Mask

训练和部署均在原始图像 resize 前将背景置黑，不修改数据集文件。默认只处理头部相机，腕部图像不变。

- `union_mask.png`：768×480 二值图，白色保留、黑色遮掉，推荐使用。
- `union_ranges.json`：同一 mask 的逐行保留区间，也可直接使用。

Mask 来自 `1019-data-v1-sidepick` 的 20 帧 SAM 分割并集，保留约 75.29% 像素。输入必须为 **768×480（宽×高）且相机视野一致**；尺寸不符会报错。

以下命令均在 `lingbot-vla` 根目录、已激活训练环境后执行。

## VLA 训练

在 `configs/vla/package_multi_posttrain.yaml` 的 `data` 段设置：

```yaml
data:
  image_mask_enabled: true
  image_mask_path: mask/data-v1/union_mask.png
  image_mask_camera: observation.images.cam_high_rgb
```

保留原有数据路径等配置，然后启动：

```bash
export PYTHONPATH="$PWD"
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/package_multi_posttrain.yaml
```

也可通过命令行覆盖开关和文件：

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/package_multi_posttrain.yaml \
  --data.image_mask_enabled true \
  --data.image_mask_path mask/data-v1/union_mask.png
```

默认关闭；设为 `false` 即可关闭。相对路径以 `lingbot-vla` 根目录为准，也支持绝对路径。相机参数使用原始字段名 `observation.images.cam_high_rgb`。

## 部署

部署默认继承 checkpoint 中 `lingbotvla_cli.yaml` 保存的 mask 配置，对实时输入执行相同掩码处理：

```bash
python -m deploy.lingbot_vla_policy --model_path /path/to/hf_ckpt
```

显式开启或更换 mask 文件：

```bash
python -m deploy.lingbot_vla_policy --model_path /path/to/hf_ckpt \
  --image-mask-enabled \
  --image-mask-path mask/data-v1/union_mask.png \
  --image-mask-camera observation.images.cam_high_rgb
```

显式关闭：

```bash
python -m deploy.lingbot_vla_policy --model_path /path/to/hf_ckpt \
  --no-image-mask-enabled
```

训练和部署应使用相同的 mask。文件不打包进 checkpoint，换机器时需一起复制；路径也可换为 `mask/data-v1/union_ranges.json`。只指定文件路径不会自动启用 mask。
