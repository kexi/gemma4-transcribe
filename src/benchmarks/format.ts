import { MODELS } from '../models.ts';

/** 値が無いとき（未計測・項目なし）の表示。 */
export const MISSING = '—';

/** CER（割合）を % 表示に丸めたときの小数桁数。表・グラフ・結論の 1 文で揃える。 */
export const PERCENT_DIGITS = 1;

/** 割合を % 単位の数字の文字列にする（記号なし）。表示と最良判定の比較キーの両方がこれを通る。 */
const percentDigitsText = (fraction: number, digits: number): string => (fraction * 100).toFixed(digits);

/** 割合（0.123）を「12.3%」にする。 */
export function formatPercent(fraction: number | null, digits = PERCENT_DIGITS): string {
  if (fraction === null) return MISSING;
  return `${percentDigitsText(fraction, digits)}%`;
}

/** 固定小数（RTF・秒など）。 */
export function formatFixed(value: number | null, digits: number): string {
  if (value === null) return MISSING;
  return value.toFixed(digits);
}

/**
 * 最良判定の比較キー：割合を formatPercent と同じ丸めで % の数値にする（0.0215 →「2.1%」→ 2.1）。
 * Why not Math.round(value * 10 ** digits): toFixed と丸め方が違い（0.0215 は toFixed で 2.1、Math.round では 2.2 相当）、
 * 画面では同じ「2.1%」なのに片方だけ太字になる食い違いが起きるため
 */
export const percentKey = (fraction: number): number => Number(percentDigitsText(fraction, PERCENT_DIGITS));

/** 最良判定の比較キー：formatFixed と同じ丸めの数値にする。 */
export const fixedKey =
  (digits: number) =>
  (value: number): number =>
    Number(value.toFixed(digits));

/** 件数。整数で来る前提だが、念のため丸めて桁区切りを付ける。 */
export function formatCount(value: number | null): string {
  if (value === null) return MISSING;
  return Math.round(value).toLocaleString('ja-JP');
}

/** git の commit SHA を 7 桁に縮める。SHA でない文字列はそのまま返す。 */
export function shortCommit(commit: string): string {
  const isSha = /^[0-9a-f]{8,40}$/i.test(commit);
  return isSha ? commit.slice(0, 7) : commit;
}

/**
 * グラフの行ラベル用の短い名前。「Gemma 4 E2B QAT mobile（q2f16）」→「E2B QAT mobile」。
 * Why not 表示名をハードコードする: 名前は JSON の label だけを正とし、モデルが増えても同じ規則で縮められるようにするため。
 * 全名は凡例・表・ツールチップに出す
 */
export function shortModelLabel(label: string): string {
  const withoutQuantization = withoutFamily(label).replace(/\s*[（(][^）)]*[）)]\s*$/, '');
  const shortened = withoutQuantization.trim();
  return shortened === '' ? label : shortened;
}

/** 「Gemma 4 E2B（q4f16）」→「E2B（q4f16）」。量子化の括弧は残す。 */
function withoutFamily(label: string): string {
  const shortened = label.replace(/^Gemma\s*4\s+/i, '').trim();
  return shortened === '' ? label : shortened;
}

/**
 * 複数モデルの短いラベル（入力と同じ順）。縮めると別のモデルと同じ名前になるもの（量子化違いの同じモデルなど）は
 * 量子化の括弧を残し、それでも重なれば元のラベルのまま出す。
 * Why not 重なっても縮めたままにする: グラフの行が色だけで区別されることになり、色覚多様性の下で読み分けられないため
 */
export function shortModelLabels(labels: readonly string[]): string[] {
  const distinct = [...new Set(labels)];
  const levels = [shortModelLabel, withoutFamily];
  const chosen = new Map<string, string>();
  for (const label of distinct) {
    const level = levels.find((shorten) => {
      const candidate = shorten(label);
      return distinct.every((other) => other === label || shorten(other) !== candidate);
    });
    chosen.set(label, level === undefined ? label : level(label));
  }
  // 段階の違う短縮同士（片方の短い名前と別のモデルの括弧付きの名前）がたまたま重なったときも元のラベルに戻す
  const counts = new Map<string, number>();
  for (const short of chosen.values()) counts.set(short, (counts.get(short) ?? 0) + 1);
  return labels.map((label) => {
    const short = chosen.get(label) ?? label;
    const isCollision = (counts.get(short) ?? 0) > 1;
    return isCollision ? label : short;
  });
}

/**
 * 表示精度で丸めた値で最小のもの（同率は全部）の添字。値が 2 つ未満なら比べる相手がいないので空。
 * 比較キー（toKey）には表示と同じ丸め（percentKey / fixedKey）を渡す。
 * 丸めてから比べるのは、画面上で同じ数字なのに片方だけ強調される食い違いを避けるため
 */
export function bestIndices(values: readonly (number | null)[], toKey: (value: number) => number): Set<number> {
  const rounded = values.map((value) => (value === null ? null : toKey(value)));
  const present = rounded.filter((value) => value !== null);
  const hasRival = present.length >= 2;
  if (!hasRival) return new Set();
  const best = Math.min(...present);
  const indices = rounded.flatMap((value, index) => (value === best ? [index] : []));
  return new Set(indices);
}

/** 検証済み参照パレットの slot 数（style.css の --bench-series-1..4 と対応）。 */
export const SERIES_SLOTS = 4;

/**
 * モデルの色 slot（1 始まり）。並び順や順位ではなく models.ts の MODELS の順（key）で固定し、
 * 絞り込みや未計測で本数が変わっても同じモデルは同じ色のままにする。
 * slot を使い切ったモデルや MODELS に無いモデルは null（中立の灰色）にする。
 * Why not 色を生成して足す: 5 色目以降は色覚多様性の下で既存の色と区別できる保証がないため（ラベルで区別する）
 */
export function seriesSlot(modelKey: string): number | null {
  const index = MODELS.findIndex((model) => model.key === modelKey);
  const hasSlot = index >= 0 && index < SERIES_SLOTS;
  return hasSlot ? index + 1 : null;
}

/** 色 slot を CSS クラス名にする（塗りは style.css のトークンで light/dark を切り替える）。 */
export function seriesClass(modelKey: string): string {
  const slot = seriesSlot(modelKey);
  return slot === null ? 'bench-series-other' : `bench-series-${slot}`;
}
