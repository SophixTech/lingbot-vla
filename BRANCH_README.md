# branch-ecot

## 目的

在独立分支开发 ECOT 标注与 LingBot-VLA 训练支持。

## 目录

- Worktree：`/home/bjtc/Sophix/worktrees/branch-ecot`
- 分支：`branch-ecot`
- 原始代码：`/home/bjtc/Sophix/lingbot-vla`（`main`）
- 数据：使用绝对路径；不复制、不修改原数据。

## 开发顺序

1. 实例数据标注验证。
2. 固定标注格式。
3. 接入数据读取。
4. 接入模型辅助监督。
5. 小规模训练与检查。

## 标注范围

只保留：`subtask`、`bbox`、已有 `grasp_label_up`。

目标不可见时：`bbox = null`。不猜框。

## Commit 规则

每次 commit 只做一类改动。提交信息写清：

```text
<type>: <change>

Data: <dataset or sample>
Check: <verification>
Rollback: <commit or file>
```

常用 `type`：`data`、`model`、`train`、`fix`、`docs`。

## 回滚

```bash
git log --oneline
git revert <commit>
```

禁止改写已共享历史。

## 已提交

- `e2dce21`：建立分支维护说明。
- `a2ade52`：统一 prompt；读取 ECOT 标签；增加三个辅助 head 和 loss。
- `8449df4`：验证两条样例、loss 掩码、旧权重加载；补训练日志。

当前仅有 episode 0、1 标注；episode 2 尚未标注（API 返回 401）。完整模型训练尚未验证。
