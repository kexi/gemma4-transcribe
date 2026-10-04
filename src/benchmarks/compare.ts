import { formatCount } from './format.ts';
import type { BenchmarkResults, ModelResult } from './schema.ts';

/** 比較（最良の強調・結論の 1 文）から外したモデルに添える言葉。 */
export const EXCLUDED_NOTE = '未完了のため比較から除外';

/**
 * 全クリップを走り終えたモデル 1 つあたりのクリップ数。datasets の n の合計を正とし、
 * どれかが欠けていれば計測済みモデルの最大の n で代える。どちらも無ければ null（件数では判定しない）。
 * Why not 計測済みモデルの n の最頻値: 2 モデルで n が違うと決められず、途中で止まったモデルが多いと誤った方を正とするため
 */
export function expectedClipCount(results: BenchmarkResults): number | null {
  const datasetCounts = results.datasets.map((dataset) => dataset.n);
  const hasAllDatasetCounts = datasetCounts.length > 0 && datasetCounts.every((n) => n !== null);
  if (hasAllDatasetCounts) return datasetCounts.reduce<number>((sum, n) => sum + (n ?? 0), 0);
  const measuredCounts = results.models.flatMap((model) => (model.overall?.n == null ? [] : [model.overall.n]));
  return measuredCounts.length > 0 ? Math.max(...measuredCounts) : null;
}

/**
 * 計測済みだが全クリップを走り終えていないモデルの状況（「未実行 12 件」など）。走り終えていれば null。
 * 未計測（overall が無い）のモデルはそもそも値が無く比較に入らないので、ここでは null を返す
 */
export function incompleteStatus(model: ModelResult, expected: number | null): string | null {
  if (model.overall === null) return null;
  const missing = model.missing ?? 0;
  if (missing > 0) return `未実行 ${formatCount(missing)} 件`;
  const n = model.overall.n;
  const isDifferentCount = expected !== null && n !== null && n !== expected;
  if (isDifferentCount) return `${formatCount(expected)} クリップ中 ${formatCount(n)} クリップ`;
  return null;
}

/**
 * 他のモデルと同じクリップ集合で測り終えたモデルだけを比較に入れる。
 * Why not 途中のモデルも含めて比べる: 易しいデータセットだけ走ったモデルが CER で「最良」に見えるなど、
 * 違うクリップ集合の平均を同じ物差しで並べることになるため
 */
export function isComparable(model: ModelResult, expected: number | null): boolean {
  return model.overall !== null && incompleteStatus(model, expected) === null;
}
