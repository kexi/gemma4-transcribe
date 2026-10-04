/**
 * 表（精度と速度・外れ値と中央値）の中身を、DOM を作る前の値として組み立てる。
 * Why not benchmarks.ts の中で DOM と一緒に作る: どのセルを太字にするか・行をどうまとめるかを
 * ブラウザ無しのテスト（vitest）で確かめられるようにするため
 */
import { EXCLUDED_NOTE, expectedClipCount, incompleteStatus, isComparable } from './compare.ts';
import { bestIndices, fixedKey, formatCount, formatFixed, formatPercent, MISSING, percentKey } from './format.ts';
import { RTF_DIGITS } from './headline.ts';
import type { BenchmarkResults, GroupMetrics, ModelResult } from './schema.ts';

export const OVERALL_LABEL = '総合';
/** 読み込み秒・ウォームアップ秒の表示桁。 */
export const SECONDS_DIGITS = 1;
/** 初回の読み込みがキャッシュからだったときの表示。ダウンロード込みの秒と並べて比べられないので値は出さない。 */
export const CACHED_FIRST_LOAD = `${MISSING}（キャッシュ済み）`;

/** 文字の列（データセット名・ID）は左寄せにするので、数値の列と区別する。 */
export type CellKind = 'number' | 'text' | 'ids';

export interface Cell {
  text: string;
  /** 列の最良値として強調する。 */
  best: boolean;
  /** 0 より大きい失敗数など、注意として色と文字で示す。 */
  warning: boolean;
  kind: CellKind;
}

export interface SummaryRow {
  model: ModelResult;
  /** 行見出しに添える計測状況（「未計測」「未実行 12 件（未完了のため比較から除外）」）。そろっていれば null。 */
  status: string | null;
  cells: Cell[];
}

export interface SummaryTable {
  headers: string[];
  rows: SummaryRow[];
}

interface Column {
  header: string;
  /** 最良判定に使う値。 */
  value: (model: ModelResult) => number | null;
  /** 表示する文字列。値だけで決まらない列（初回読み込み）はモデルを見る。 */
  format: (value: number | null, model: ModelResult) => string;
  /** 小さいほど良い列だけ最良を強調する。比較キーは表示と同じ丸めにする（percentKey / fixedKey）。 */
  bestKey?: (value: number) => number;
  warnWhenPositive?: boolean;
}

/** 行見出しに添える計測状況。未計測・未完了を言い、全部そろっていれば null。 */
export function modelStatus(model: ModelResult, expected: number | null): string | null {
  if (model.overall === null) return '未計測';
  const incomplete = incompleteStatus(model, expected);
  return incomplete === null ? null : `${incomplete}（${EXCLUDED_NOTE}）`;
}

/**
 * 初回読み込み秒の表示。ダウンロードを伴ったと分かっているときは秒、キャッシュからだったときは値を出さない。
 * 不明（first_load_downloaded が無い古い JSON）のときは秒をそのまま出す（読み込み秒の説明に「未キャッシュならダウンロードを含む」と書いてある）。
 * Why not 不明も「キャッシュ済み」と出す: 実際にはダウンロードを含んでいたかもしれない値を、事実と違う言葉で隠すことになるため
 */
export function firstLoadText(model: ModelResult): string {
  const isCachedLoad = model.firstLoadDownloaded === false;
  if (isCachedLoad) return CACHED_FIRST_LOAD;
  return formatFixed(model.loadFirstS, SECONDS_DIGITS);
}

const percent = (value: number | null): string => formatPercent(value);
const seconds = (value: number | null): string => formatFixed(value, SECONDS_DIGITS);

function summaryColumns(results: BenchmarkResults): Column[] {
  return [
    ...results.datasets.map((dataset): Column => ({
      header: `${dataset.label} 正規化`,
      value: (model) => model.perDataset[dataset.key]?.cerNorm ?? null,
      format: percent,
      bestKey: percentKey,
    })),
    {
      header: `${OVERALL_LABEL} 厳密`,
      value: (model) => model.overall?.cerStrict ?? null,
      format: percent,
      bestKey: percentKey,
    },
    {
      header: `${OVERALL_LABEL} 正規化`,
      value: (model) => model.overall?.cerNorm ?? null,
      format: percent,
      bestKey: percentKey,
    },
    {
      header: `${OVERALL_LABEL} 読み`,
      value: (model) => model.overall?.cerReading ?? null,
      format: percent,
      bestKey: percentKey,
    },
    {
      header: 'RTF',
      value: (model) => model.overall?.rtf ?? null,
      format: (value) => formatFixed(value, RTF_DIGITS),
      bestKey: fixedKey(RTF_DIGITS),
    },
    // Why not 初回も最良を強調する: キャッシュから読んだモデルとダウンロードしたモデルの秒は意味が違い、速い方が良いとは言えないため
    { header: '読み込み 初回 秒', value: (model) => model.loadFirstS, format: (_, model) => firstLoadText(model) },
    {
      header: '読み込み キャッシュ 秒',
      value: (model) => model.loadCachedS,
      format: seconds,
      bestKey: fixedKey(SECONDS_DIGITS),
    },
    {
      header: 'ウォームアップ 秒',
      value: (model) => model.warmupS,
      format: seconds,
      bestKey: fixedKey(SECONDS_DIGITS),
    },
    { header: '失敗', value: (model) => model.overall?.failed ?? null, format: formatCount, warnWhenPositive: true },
  ];
}

/**
 * 精度と速度の表。最良の強調は、全クリップを走り終えたモデルの間だけで決める
 * （未完了のモデルの値は null とみなして比較から外し、行見出しにその旨を書く）。
 */
export function summaryTable(results: BenchmarkResults): SummaryTable {
  const columns = summaryColumns(results);
  const expected = expectedClipCount(results);
  const comparable = results.models.map((model) => isComparable(model, expected));
  const best = columns.map((column) => {
    if (column.bestKey === undefined) return new Set<number>();
    const values = results.models.map((model, index) => (comparable[index] === true ? column.value(model) : null));
    return bestIndices(values, column.bestKey);
  });
  const rows = results.models.map((model, rowIndex): SummaryRow => ({
    model,
    status: modelStatus(model, expected),
    cells: columns.map((column, columnIndex): Cell => {
      const value = column.value(model);
      return {
        text: column.format(value, model),
        best: best[columnIndex]?.has(rowIndex) ?? false,
        warning: column.warnWhenPositive === true && value !== null && value > 0,
        kind: 'number',
      };
    }),
  }));
  return { headers: columns.map((column) => column.header), rows };
}

/** 内訳表の 1 行（モデル × データセット、または総合）。 */
export interface BreakdownRow {
  isOverall: boolean;
  cells: Cell[];
}

/** 内訳表の 1 モデル分。1 モデル = 1 つの tbody にし、行見出し（scope=rowgroup）の範囲をそのモデルに限る。 */
export interface BreakdownGroup {
  model: ModelResult;
  rows: BreakdownRow[];
}

export function breakdownHeaders(results: BenchmarkResults): { text: string; kind: CellKind }[] {
  return [
    { text: 'モデル', kind: 'text' },
    { text: 'データセット', kind: 'text' },
    { text: '件数', kind: 'number' },
    { text: '失敗', kind: 'number' },
    { text: 'CER 正規化', kind: 'number' },
    { text: '中央値', kind: 'number' },
    { text: `外れ値（>${formatPercent(results.outlierThreshold, 0)}）`, kind: 'number' },
    { text: '外れ値除外 CER（参考値）', kind: 'number' },
    { text: 'RTF', kind: 'number' },
    { text: '誤りの大きい例（ID）', kind: 'ids' },
  ];
}

/** グループ（データセット key、または総合 = null）の指標を引く。 */
export const groupOf = (model: ModelResult, group: string | null): GroupMetrics | null =>
  group === null ? model.overall : (model.perDataset[group] ?? null);

const plain = (text: string, kind: CellKind = 'number'): Cell => ({ text, best: false, warning: false, kind });

/** 外れ値と中央値の表。モデルごとに、各データセットの行と総合の行をまとめる。 */
export function breakdownGroups(results: BenchmarkResults): BreakdownGroup[] {
  const groups: { key: string | null; label: string }[] = [
    ...results.datasets.map((dataset) => ({ key: dataset.key, label: dataset.label })),
    { key: null, label: OVERALL_LABEL },
  ];
  return results.models.map((model) => ({
    model,
    rows: groups.map(({ key, label }): BreakdownRow => {
      const metrics = groupOf(model, key);
      const worst = key === null ? [] : (model.worstIds[key] ?? []);
      const failed = metrics?.failed ?? null;
      const hasFailed = failed !== null && failed > 0;
      return {
        isOverall: key === null,
        cells: [
          plain(label, 'text'),
          plain(formatCount(metrics?.n ?? null)),
          { ...plain(formatCount(failed)), warning: hasFailed },
          plain(formatPercent(metrics?.cerNorm ?? null)),
          plain(formatPercent(metrics?.cerNormMedian ?? null)),
          plain(formatCount(metrics?.outliers ?? null)),
          plain(formatPercent(metrics?.cerNormExclOutliers ?? null)),
          plain(formatFixed(metrics?.rtf ?? null, RTF_DIGITS)),
          plain(worst.length > 0 ? worst.join(', ') : MISSING, 'ids'),
        ],
      };
    }),
  }));
}

/**
 * run 全体で外れ値として記録されたクリップ数を言う 1 文（JSON の outlier_ids）。項目が無い古い JSON では空。
 * Why not モデルごとの外れ値数を足す: 同じクリップが複数のモデルで外れ値になると重複して数えるため、run で 1 回だけ出す
 */
export function runOutlierSentence(results: BenchmarkResults): string {
  if (results.outlierIds === null) return '';
  return `この実行で外れ値として記録されたクリップは計 ${formatCount(results.outlierIds.length)} 件です（ID は集計 JSON の outlier_ids）。`;
}
