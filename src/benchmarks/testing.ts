/** テスト用の組み立て関数（本番のページからは読み込まない）。 */
import fixture from './fixture.json' with { type: 'json' };
import { parseResults, type BenchmarkResults, type GroupMetrics, type ModelResult } from './schema.ts';

/** フィクスチャを読んだ結果。 */
export const fixtureResults = (): BenchmarkResults => {
  const result = parseResults(fixture);
  if (!result.ok) throw new Error(result.errors.join('\n'));
  return result.data;
};

/** 総合の指標。件数はフィクスチャのデータセット合計（300）に合わせる。 */
export const metrics = (
  cerNorm: number | null,
  rtf: number | null,
  options: { failed?: number; n?: number | null } = {},
): GroupMetrics => ({
  n: options.n === undefined ? 300 : options.n,
  failed: options.failed ?? 0,
  cerStrict: null,
  cerNorm,
  cerReading: null,
  rtf,
  cerNormMedian: null,
  outliers: null,
  cerNormExclOutliers: null,
});

export const model = (
  key: string,
  label: string,
  overall: GroupMetrics | null,
  overrides: Partial<ModelResult> = {},
): ModelResult => ({
  key,
  label,
  id: null,
  revision: null,
  loadFirstS: null,
  firstLoadDownloaded: null,
  loadCachedS: null,
  warmupS: null,
  missing: overall === null ? null : 0,
  perDataset: {},
  overall,
  worstIds: {},
  ...overrides,
});

/** フィクスチャの datasets（合計 300 クリップ）のまま、models だけ差し替える。 */
export const withModels = (models: ModelResult[]): BenchmarkResults => ({ ...fixtureResults(), models });
