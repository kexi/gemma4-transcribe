import { EXCLUDED_NOTE, expectedClipCount, isComparable } from './compare.ts';
import { bestIndices, fixedKey, formatCount, formatFixed, formatPercent, percentKey } from './format.ts';
import type { BenchmarkResults, ModelResult } from './schema.ts';

/** RTF の表示桁。表と結論の 1 文で揃える。 */
export const RTF_DIGITS = 3;

const NO_MEASURED = 'まだ計測済みのモデルがありません。';

/** 直前の文字が半角英数字（と半角の閉じ括弧）なら、続く語との間に半角空白を挟む。 */
const endsWithAscii = (text: string): boolean => /[A-Za-z0-9)\]]$/.test(text);
const startsWithAscii = (text: string): boolean => /^[A-Za-z0-9([]/.test(text);

/**
 * 名前の後に助詞を続ける。英数字で終わる名前だけ半角空白を挟む（「E2B が」）。
 * 全角の括弧で終わる名前（「E2B（q4f16）が」）に空白を挟むと間延びするため挟まない
 */
export const withParticle = (name: string, particle: string): string =>
  endsWithAscii(name) ? `${name} ${particle}` : `${name}${particle}`;

/**
 * 複数の名前を「と」でつなぐ。「と」の前は withParticle と同じ規則、後ろは次の名前が英数字で始まるときだけ空白を挟む。
 * Why not 常に「 と 」でつなぐ: 「E2B（q4f16） と」のように全角の括弧の後に空白が入り、withParticle の規則と食い違うため
 */
export const joinLabels = (labels: readonly string[]): string =>
  labels.reduce((joined, label) => {
    const head = withParticle(joined, 'と');
    return startsWithAscii(label) ? `${head} ${label}` : `${head}${label}`;
  });

const labelsOf = (models: readonly ModelResult[]): string => joinLabels(models.map((model) => model.label));

/** 値が失敗クリップを除いたものであることの注記。1 モデルなら件数、同率の複数モデルならまとめて言う。 */
const failedNote = (models: readonly ModelResult[]): string => {
  const failedCounts = models.map((model) => model.overall?.failed ?? 0);
  const hasFailed = failedCounts.some((failed) => failed > 0);
  if (!hasFailed) return '';
  const [onlyFailed] = failedCounts;
  const isSingle = models.length === 1 && onlyFailed !== undefined;
  return isSingle ? `、失敗 ${formatCount(onlyFailed)} 件を除く` : '、失敗したクリップを除く';
};

/** 比較から外したモデルを言う 1 文。外したモデルが無ければ空。 */
const exclusionSentence = (excluded: readonly ModelResult[]): string =>
  excluded.length === 0 ? '' : `${withParticle(labelsOf(excluded), 'は')}${EXCLUDED_NOTE}しています。`;

/**
 * ページ先頭の結論の 1 文を JSON の数値から作る。
 * 精度（総合 CER・正規化）の最良と、速度（RTF）の最速を言う。同率は並べて言い、勝者を 1 つに決め打ちしない。
 * 全クリップを走り終えていないモデル（未実行がある・件数が違う）は比べず、外したことを別の 1 文で言う。
 * Why not 文面を手で書く: 結果を更新するたびに文と表が食い違う恐れがあるため、数値と同じ JSON から毎回作る
 */
export function buildHeadline(results: BenchmarkResults): string {
  const measured = results.models.filter((model) => model.overall?.cerNorm != null);
  if (measured.length === 0) return NO_MEASURED;

  const expected = expectedClipCount(results);
  const compared = measured.filter((model) => isComparable(model, expected));
  const excluded = measured.filter((model) => !isComparable(model, expected));
  const exclusion = exclusionSentence(excluded);
  if (compared.length === 0) return `全クリップの計測を終えたモデルはまだありません。${exclusion}`;

  const cerValues = compared.map((model) => model.overall?.cerNorm ?? null);
  if (compared.length === 1) {
    const [only] = compared;
    if (only === undefined) return NO_MEASURED;
    const n = only.overall?.n;
    const clips = n == null ? '' : `（${formatCount(n)} クリップ${failedNote([only])}）`;
    const rivals = exclusion === '' ? '比較対象のモデルはまだ計測されていません。' : exclusion;
    return `日本語の総合 CER（正規化）は ${withParticle(only.label, 'で')} ${formatPercent(cerValues[0] ?? null)}${clips}。${rivals}`;
  }

  const bestCer = bestIndices(cerValues, percentKey);
  const winners = compared.filter((_, index) => bestCer.has(index));
  const [firstWinner] = winners;
  if (firstWinner === undefined) return NO_MEASURED;
  const winnerCer = formatPercent(firstWinner.overall?.cerNorm ?? null);
  const isTie = winners.length > 1;
  const accuracy = isTie
    ? `日本語の総合 CER（正規化）は ${withParticle(labelsOf(winners), 'が')}同率で最良（${winnerCer}${failedNote(winners)}）`
    : `日本語の総合 CER（正規化）は ${withParticle(firstWinner.label, 'が')}最良（${winnerCer}${failedNote(winners)}）`;

  const rtfValues = compared.map((model) => model.overall?.rtf ?? null);
  const fastest = [...bestIndices(rtfValues, fixedKey(RTF_DIGITS))]
    .map((index) => compared[index])
    .filter((model) => model !== undefined);
  const [firstFastest] = fastest;
  if (firstFastest === undefined) return `${accuracy}。${exclusion}`;
  const fastestRtf = formatFixed(firstFastest.overall?.rtf ?? null, RTF_DIGITS);
  const isSameAsWinner = !isTie && fastest.length === 1 && firstFastest.key === firstWinner.key;
  if (isSameAsWinner) return `${accuracy}で、速度も最速（RTF ${fastestRtf}）。${exclusion}`;
  return `${accuracy}、速度は ${withParticle(labelsOf(fastest), 'が')}最速（RTF ${fastestRtf}${failedNote(fastest)}）。${exclusion}`;
}
