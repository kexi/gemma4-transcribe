import { MODELS, type ModelDtype } from '../models.ts';
import { LANGUAGES } from '../prompt.ts';

/**
 * 評価ページと Playwright（eval/gemma4_eval/run_browser.py）・prepare.py との契約。
 * 形を変えるときは Python 側も合わせて変える。
 */

/** `eval/data/manifest.json` の 1 件。 */
export interface ManifestItem {
  id: string;
  dataset: string;
  /** manifest.json からの相対パス。 */
  path: string;
  duration_s: number;
  reference: string;
}

export interface Manifest {
  items: ManifestItem[];
}

/** 1 クリップの結果。完了したものから `window.__eval.results` に push する。 */
export interface EvalResult {
  id: string;
  dataset: string;
  reference: string;
  hypothesis: string;
  /** manifest の値ではなくデコード後の実長（秒）。 */
  durationS: number;
  /** fetch + decodeToMono16k（ミリ秒）。 */
  decodeMs: number;
  /** 全区間の transcribe 合計（Worker への送信から done まで、ミリ秒）。 */
  inferMs: number;
  segments: number;
  /** 個別失敗は記録して続行する。 */
  error: string | null;
}

/** 評価したモデルの素性（models.ts の値）。集計 JSON に載せ、どの revision・量子化の結果かを残す。 */
export interface EvalModelInfo {
  key: string;
  label: string;
  id: string;
  revision: string;
  dtype: ModelDtype;
}

/** `window.__eval` として公開する評価の進行状況。Playwright はこれをポーリングする。 */
export interface EvalState {
  /** warming-up: 読み込み後、計測しない文字起こしを 1 回流している間（シェーダのコンパイル等を計測から外す）。 */
  phase: 'loading-model' | 'warming-up' | 'running' | 'done' | 'error';
  model: string;
  modelInfo: EvalModelInfo | null;
  /** このページでのモデルの読み込み時間（Worker への load 送信から loaded まで）。 */
  loadMs: number | null;
  /** 計測しないウォームアップ 1 回の所要時間。対象クリップが無い（limit=0）ときは null のまま。 */
  warmupMs: number | null;
  /**
   * 処理中（ウォームアップを含む）のクリップ ID。結果を results に push した時点で null に戻す。
   * ページがクラッシュしたとき、読み取り済みの最後の結果ではなく、実際に処理中だったクリップを特定するため
   */
  currentId: string | null;
  /**
   * Cache API にあるこのモデル（id + revision）のファイル数。読み込みの直前（before）と直後（after）に数える。
   * run_browser は after が before より増えていれば「このページでダウンロードした」と判断する。数えられなければ null
   */
  cachedModelFiles: { before: number | null; after: number | null };
  total: number;
  completed: number;
  results: EvalResult[];
  error: string | null;
}

export interface EvalParams {
  /** models.ts の key。 */
  model: string;
  /** manifest.json の URL（ページからの相対も可）。 */
  manifest: string;
  /** 評価する件数の上限。未指定なら offset 以降をすべて。 */
  limit: number | null;
  offset: number;
  /** プロンプトに埋め込む英語の言語名（prompt.ts の LANGUAGES の name）。 */
  language: string;
}

export const DEFAULT_MANIFEST_URL = '/data/manifest.json';
export const DEFAULT_LANGUAGE = 'Japanese';

/** 空文字は未指定として扱う（just の `limit=""` がそのまま `limit=` で渡ってくるため）。 */
function parseCount(name: string, raw: string | null): number | null {
  const isUnset = raw === null || raw.trim() === '';
  if (isUnset) return null;
  const isNonNegativeInteger = /^\d+$/.test(raw.trim());
  if (!isNonNegativeInteger) throw new Error(`${name} は 0 以上の整数で指定してください: ${raw}`);
  return Number(raw.trim());
}

/**
 * URL のクエリ文字列から評価条件を読む。
 * 不明なモデルや言語は findModel のように既定へフォールバックせず失敗させる。
 * 黙って別のモデル・プロンプトで測ると、結果ファイルの名前と中身が食い違うため
 */
export function parseEvalParams(search: string): EvalParams {
  const params = new URLSearchParams(search);

  const model = params.get('model');
  const isKnownModel = MODELS.some((option) => option.key === model);
  if (model === null || !isKnownModel) {
    const keys = MODELS.map((option) => option.key).join(', ');
    throw new Error(`model には次のいずれかを指定してください: ${keys}（指定値: ${model ?? '未指定'}）`);
  }

  const language = params.get('language') || DEFAULT_LANGUAGE;
  const isKnownLanguage = LANGUAGES.some((option) => option.name === language);
  if (!isKnownLanguage) {
    const names = LANGUAGES.map((option) => option.name).join(', ');
    throw new Error(`language には次のいずれかを指定してください: ${names}（指定値: ${language}）`);
  }

  return {
    model,
    manifest: params.get('manifest') || DEFAULT_MANIFEST_URL,
    limit: parseCount('limit', params.get('limit')),
    offset: parseCount('offset', params.get('offset')) ?? 0,
    language,
  };
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

function parseManifestItem(value: unknown, index: number): ManifestItem {
  if (!isRecord(value)) throw new Error(`manifest.items[${index}] がオブジェクトではありません`);
  const { id, dataset, path, duration_s: durationS, reference } = value;
  const hasValidFields =
    typeof id === 'string' &&
    typeof dataset === 'string' &&
    typeof path === 'string' &&
    typeof durationS === 'number' &&
    typeof reference === 'string';
  if (!hasValidFields) throw new Error(`manifest.items[${index}] の形が契約と違います`);
  return { id, dataset, path, duration_s: durationS, reference };
}

/**
 * manifest.json の中身を検証する。
 * 形の崩れを最初に検出しないと、数千件の推論が終わってから score.py で欠損に気付くことになるため、読み込み時点で落とす
 */
export function parseManifest(value: unknown): Manifest {
  const hasItems = isRecord(value) && Array.isArray(value.items);
  if (!hasItems) throw new Error('manifest に items 配列がありません');
  return { items: (value.items as unknown[]).map(parseManifestItem) };
}

/** offset から limit 件を取り出す（limit が null なら末尾まで）。 */
export function selectItems<T>(items: readonly T[], offset: number, limit: number | null): T[] {
  const end = limit === null ? undefined : offset + limit;
  return items.slice(offset, end);
}
