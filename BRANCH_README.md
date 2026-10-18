# branch-ecot-v2 本次改动记录

## 2026-10-18：新增逐层标签 K/V 条件

### 建立分支

- 目录：`/home/bjtc/Sophix/worktrees/branch-ecot-v2`
- Git 分支：`branch-ecot-v2`，起点 `00cf5c8`。
- 从 `branch-ecot` 当前工作树复制代码、未提交改动、文档和输出文件；源目录未修改。

### 模型与配置修改

- 新增 `EcotLabelKVProjector`，将 VLM 预测的 label 概率转换成 Action Expert 每层独立的 K/V。
- 沿用 Action Expert 的 Q 读取标签 K/V；训练中屏蔽图文前缀 query 对标签 K/V 的访问。
- 真实标签只作分类监督，动作条件使用预测概率；动作损失可回传到分类头和 K/V 生成模块。
- 推理只运行一次 VLM，标签 K/V 一次生成、多步去噪复用。训练仍采用源分支的两遍前缀计算。
- 关闭 `ecot_action_conditioning`，启用 `ecot_label_kv_conditioning`，新输出使用 `output/ecot1003_label_kv_v2/`。
- 配套更新配置传递、LoRA 解冻和新增模块权重初始化。
- 用中文分支说明替换原 LingBot 项目 README；本记录只保留此次工作。

### 验证与当前状态

- `py_compile` 与 `git diff --check` 通过。
- ECOT 全部 7 项单元测试通过，覆盖 K/V 形状与梯度、真实 attention 隔离、推理缓存构建与复用。
- 本次代码、测试及文档整理提交，发布到 `SophixTech/lingbot-vla` 的 `zyh-ecot-v2` 分支；本地工作分支保留 `branch-ecot-v2`。
- 模型权重、训练输出、原始数据与生成的视频/图表不纳入此次提交。

### 本次 GPU 短训练与部署通路检查

- 日期：2026-10-18。以 609 条 `sampled_audit_passed` 标注、406455 帧构建训练池，仅抽样执行 2 个训练步（batch 1），未遍历全部数据。
- 核查源 episode：既有 9 条排除数据及已指出错标的 539 均不在该标注子集中。
- 总损失：0.4467、0.6046；动作损失：0.2572、0.2591；梯度范数：4.3479、6.4395。训练正常结束。
- 保存目录：`output/ecot_label_kv_smoke_20261018/checkpoints/global_step_2/hf_ckpt/`。
- 部署加载器严格重载通过；真实观测执行 10 步去噪，输出形状 `[1, 50, 75]`（含模型填充维度），全部有限。
- 实测单次观测的 VLM 前缀与标签 K/V 生成器均只调用一次。
- 独立动作损失反传：标签分类头梯度范数约 0.003905，全部 36 层 K/V 投影梯度非零且有限。
- 新增离线复检脚本 `scripts/check_label_kv_smoke.py`；结果保存到 `output/ecot_label_kv_smoke_20261018/smoke_validation.json`。
- 仅验证模型通路，不代表行为效果；未连接机器人，未启动长训练。

结构、配置及训练检查命令见 [README.md](README.md)。
