# ECOT 逐层标签 K/V（branch-ecot-v2）

本分支新增标签条件通路：将 VLM 预测的 label 概率编码成 Action Expert 每层可读取的 K/V，让标签参与各层注意力计算。

## 当前 LEMON 训练入口

本次 8 卡 H200 训练使用 [configs/vla/package_multi_posttrain.yaml](configs/vla/package_multi_posttrain.yaml)。从本项目根目录启动，训练环境需能访问配置中的模型、数据、标注和归一化文件路径：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  tasks/vla/train_lingbotvla.py configs/vla/package_multi_posttrain.yaml
```

该配置仅使用有 ECOT 标注的训练 episode，启用辅助监督和预测标签 K/V；保留本次选定的训练超参数。下文 `ecot1003.yaml` 是原分支 LoRA 短测参考配置，历史验证结果不代表当前 8 卡配置已完成运行验证。

已清理旧 mixed/separated 实验的 5 个训练 YAML、11 个启动/监控 shell 脚本及 5 个 systemd 服务/定时器文件；它们固定指向旧工程和实验输出。文件原版本保留在 Git 历史中。通用工具、评估脚本、测试和其余参考配置保留。

## 数据流

1. 图像与固定 prompt 输入 VLM，生成图文前缀特征和各层 K/V 缓存。
2. 标签分类头根据最终前缀特征预测概率：类别 0 为 label-down，类别 1 为 label-up。
3. 小型 MLP 编码概率，各层独立投影生成标签 K/V。
4. Action Expert 使用现有动作 Q 读取图文、标签及动作自身的 K/V，预测 velocity 并迭代去噪。

推理每次观测只运行一次 VLM，标签 K/V 生成一次后加入缓存，供全部去噪步复用。

训练仍沿用源分支的两遍前缀计算：先预测标签，再执行带标签 K/V 的前缀与动作后缀联合前向。图文前缀 query 不能读取标签 K/V，state/action 后缀 query 可以读取。

真实标签只参与分类监督，不作为动作条件输入。动作损失可回传到标签分类头、标签编码器及各层 K/V 投影。这是学习得到的标签表示，不是 VLM 编码 prompt 得到的 K/V；是否改善行为需要训练后验证。

## 本次实现

- 新增 `EcotLabelKVProjector`：将预测概率编码并投影成各层标签 K/V。
- 扩展 `QwenvlWithExpertModel.forward`：支持额外 K/V 和动作后缀访问掩码。
- 为标签预留 RoPE 位置，保持训练与推理的位置约定一致。
- 推理缓存仅追加一次标签 K/V，多步去噪复用。
- 配套更新配置传递、LoRA 解冻及新增模块权重初始化。
- 关闭原有 10/11 维 ECOT 加法条件，保留 subtask、bbox、label、visibility 辅助监督。

## 配置与数据

配置：[configs/vla/ecot1003.yaml](configs/vla/ecot1003.yaml)。沿用源分支的数据、固定 prompt、`bounds_99` 归一化和 LoRA 训练方案。

```yaml
ecot_action_conditioning: false
ecot_label_kv_conditioning: true
```

- 训练数据：`/home/bjtc/Sophix/datasets/marked_9_1_0922/train/`
- ECOT 标注：`/home/bjtc/Sophix/datasets/ecot_annotations/ecot1003/`
- 归一化：`/home/bjtc/Sophix/datasets/marked_9_1_0922/train_stats/norm_stats.json`
- 新训练输出：`/home/bjtc/Sophix/worktrees/branch-ecot-v2/output/ecot1003_label_kv_v2/`

复制的 `output/` 包含源分支已有结果，不是 v2 的训练结果。

## 训练通路检查

以下命令用于检查 2 步训练与保存通路；已完成一次检查，实测输出目录为 `output/ecot_label_kv_smoke_20261018/`：

```bash
cd /home/bjtc/Sophix/worktrees/branch-ecot-v2
PYTHONPATH=. OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
/home/bjtc/Sophix/lingbot-vla/.venv/bin/torchrun \
  --nnodes=1 --nproc-per-node=1 --master-addr=127.0.0.1 --master-port=29652 \
  tasks/vla/train_lingbotvla.py configs/vla/ecot1003.yaml \
  --train.max_steps=2 --train.gradient_accumulation_steps=1 --train.save_steps=2 \
  --train.output_dir=/home/bjtc/Sophix/worktrees/branch-ecot-v2/output/ecot_label_kv_smoke
```

## 验证状态

- 修改模块的 Python 编译检查通过。
- ECOT 全部 7 项单元测试通过，覆盖 K/V 形状与梯度、真实 attention 隔离、推理缓存构建与复用。
- 2026-10-18：已在真实模型和 GPU 上完成 2 步训练、checkpoint 保存、部署严格重载及真实数据 10 步去噪。609 条已审核标注共 406455 帧作为数据池；短测仅抽样 2 个训练步，没有遍历全部数据。
- 两步总损失分别为 0.4467、0.6046，梯度范数分别为 4.3479、6.4395，均有限。
- 动作损失单独反传时，标签分类头和 36 层 K/V 投影均有非零有限梯度；推理实测每次观测只调用一次 VLM 前缀和标签 K/V 生成器。
- 未进行正式长训练或行为效果评测，不能据此确认翻面行为改善。全过程离线，未连接机器人。
- 实测结果：`output/ecot_label_kv_smoke_20261018/smoke_validation.json`；复检脚本：`scripts/check_label_kv_smoke.py`。

本次改动记录见 [BRANCH_README.md](BRANCH_README.md)。
