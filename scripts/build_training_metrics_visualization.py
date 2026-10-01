from __future__ import annotations

import json
from pathlib import Path


ROOT = Path('/home/bjtc/Sophix/lingbot-vla/output')
OUT = Path('/home/bjtc/.codex/visualizations/2026/09/23/01a0cd12-2c6c-7e80-ab36-9005da3ef7a8/training-metrics-comparison.html')
WINDOW = 100

RUNS = [
    ('清理前\nmarked_full0919', 'marked_full0919', ROOT / 'marked_full0919/checkpoints/loss.jsonl'),
    ('清理后\nmarked_9_1_0922', 'marked_9_1_0922', ROOT / 'marked_9_1_0922/checkpoints/loss.jsonl'),
    ('分开 label-up', 'label_up', ROOT / 'separated_model/label_up/checkpoints/loss.jsonl'),
    ('分开 label-down', 'label_down', ROOT / 'separated_model/label_down/checkpoints/loss.jsonl'),
    ('混合 + prompt', 'mixed_conditioned', ROOT / 'separated_model/mixed_conditioned/checkpoints/loss.jsonl'),
]

METRICS = {
    'marked_full0919': ROOT / 'marked_full0919/evaluation_mse_20260921/metrics.json',
    'label_up': ROOT / 'separated_model/label_up/test_metrics/metrics.json',
    'label_down': ROOT / 'separated_model/label_down/test_metrics/metrics.json',
    'mixed_conditioned': ROOT / 'separated_model/mixed_conditioned/test_metrics/metrics.json',
}


def load_loss(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_step = {int(row['step']): row for row in rows}
    rows = [by_step[step] for step in sorted(by_step)]
    values = [float(row['loss']) for row in rows]
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - WINDOW + 1)
        smoothed.append(sum(values[start:idx + 1]) / (idx - start + 1))
    return [(int(row['step']), value) for row, value in zip(rows, smoothed)]


def load_metric(path: Path):
    data = json.loads(path.read_text())
    if 'results' in data:
        return data['results']['regular_64_episodes']
    return data['metrics']


def esc(value: str) -> str:
    return (value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;'))


def line_chart(series, width=900, height=360):
    left, right, top, bottom = 78, 870, 38, 300
    xmax = max(point[0] for points in series.values() for point in points)
    ymax = 0.45
    def x(value):
        return left + value / xmax * (right - left)
    def y(value):
        return bottom - value / ymax * (bottom - top)
    colors = ['var(--viz-series-1)', 'var(--viz-series-2)', 'var(--viz-series-3)',
              'var(--viz-series-4)', 'var(--viz-series-5)']
    parts = [f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="100步滑动平均训练 Loss 对比">',
             '<title>100步滑动平均训练 Loss 对比</title>',
             '<text class="title" x="450" y="20" text-anchor="middle">训练 Loss（100 步滑动平均）</text>']
    for tick in [0, .1, .2, .3, .4]:
        yy = y(tick)
        parts += [f'<line class="grid" x1="{left}" x2="{right}" y1="{yy:.1f}" y2="{yy:.1f}"/>',
                  f'<text class="tick" x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end">{tick:.1f}</text>']
    for tick in [0, 2000, 4000, 6000, 8000, 10000]:
        xx = x(tick)
        parts += [f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{bottom}"/>',
                  f'<text class="tick" x="{xx:.1f}" y="{bottom + 20}" text-anchor="middle">{tick}</text>']
    parts += [f'<rect class="frame" x="{left}" y="{top}" width="{right-left}" height="{bottom-top}"/>']
    for idx, (label, points) in enumerate(series.items()):
        stride = max(1, len(points) // 500)
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        coords = ' '.join(f'{x(step):.2f},{y(value):.2f}' for step, value in sampled)
        parts.append(f'<polyline class="line" style="stroke:{colors[idx]}" points="{coords}"/>')
    parts += ['<text class="axis-title" x="474" y="350" text-anchor="middle">训练 step</text>',
              '<text class="axis-title" transform="translate(18 170) rotate(-90)" text-anchor="middle">loss</text>']
    for idx, label in enumerate(series):
        xx = left + (idx % 3) * 265
        yy = 324 + (idx // 3) * 20
        parts += [f'<line x1="{xx}" x2="{xx + 22}" y1="{yy}" y2="{yy}" style="stroke:{colors[idx]};stroke-width:2"/>',
                  f'<text class="legend-text" x="{xx + 28}" y="{yy + 4}">{esc(label.replace(chr(10), " / "))}</text>']
    parts.append('</svg>')
    return ''.join(parts)


def bar_chart(title, metric_key, label, metric_max, width=900, height=360):
    left, right, top, bottom = 78, 870, 42, 286
    labels = [run[0] for run in RUNS]
    values = []
    for _, key, _ in RUNS:
        if key not in METRICS:
            values.append(None)
            continue
        data = load_metric(METRICS[key])
        block = (data.get(metric_key) or data.get('raw_action_16d_mixed_units')
                 or data.get('action_16d_mixed_units'))
        values.append(float(block[label]))
    gap = 18
    bar_w = ((right - left) - gap * (len(values) + 1)) / len(values)
    colors = ['var(--viz-series-1)', 'var(--viz-series-2)', 'var(--viz-series-3)',
              'var(--viz-series-4)', 'var(--viz-series-5)']
    def y(value):
        return bottom - value / metric_max * (bottom - top)
    parts = [f'<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
             f'<title>{esc(title)}</title>', f'<text class="title" x="450" y="22" text-anchor="middle">{esc(title)}</text>']
    for tick in [0, metric_max / 4, metric_max / 2, metric_max * 3 / 4, metric_max]:
        yy = y(tick)
        parts += [f'<line class="grid" x1="{left}" x2="{right}" y1="{yy:.1f}" y2="{yy:.1f}"/>',
                  f'<text class="tick" x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end">{tick:.3f}</text>']
    parts.append(f'<rect class="frame" x="{left}" y="{top}" width="{right-left}" height="{bottom-top}"/>')
    for idx, (run_label, value) in enumerate(zip(labels, values)):
        xx = left + gap + idx * (bar_w + gap)
        if value is None:
            continue
        yy = y(value)
        parts += [f'<rect class="bar" style="fill:{colors[idx]}" x="{xx:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{bottom-yy:.1f}"/>',
                  f'<text class="value" x="{xx + bar_w/2:.1f}" y="{yy - 7:.1f}" text-anchor="middle">{value:.4f}</text>',
                  f'<text class="tick" transform="translate({xx + bar_w/2:.1f} {bottom + 18}) rotate(-25)" text-anchor="end">{esc(run_label.replace(chr(10), " / "))}</text>']
    parts += [f'<text class="axis-title" x="474" y="350" text-anchor="middle">模型</text>',
              f'<text class="axis-title" transform="translate(18 165) rotate(-90)" text-anchor="middle">{esc(label)}</text>', '</svg>']
    return ''.join(parts)


def main():
    series = {label: load_loss(path) for label, _, path in RUNS}
    root = ('<div id="training-metrics-comparison"><style>'
            '#training-metrics-comparison{font-family:system-ui,sans-serif;color:var(--foreground);width:100%}'
            '.plot{width:100%;height:auto;display:block;margin:8px 0}'
            '.title,.axis-title,.tick,.legend-text,.value{fill:var(--foreground);font-size:12px}'
            '.title{font-size:14px;font-weight:500}.grid{stroke:var(--border);stroke-width:1;opacity:.55}'
            '.frame{fill:none;stroke:var(--border);stroke-width:1}.line{fill:none;stroke-width:2}'
            '.bar{opacity:.82}'
            '</style>')
    root += line_chart(series)
    root += bar_chart('统一 normalized 16D MAE', 'normalized_16d', 'mae', .24)
    root += bar_chart('统一 normalized 16D MSE', 'normalized_16d', 'mse', .125)
    root += bar_chart('统一 raw action 16D MAE', 'raw_action_16d_mixed_units', 'mae', .025)
    root += bar_chart('统一 raw action 16D MSE', 'raw_action_16d_mixed_units', 'mse', .005)
    root += '</div>'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(root + '\n')
    print(OUT)


if __name__ == '__main__':
    main()
