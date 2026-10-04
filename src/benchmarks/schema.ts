/**
 * eval/results/latest.json（eval/gemma4_eval/score.py の build_page_json の出力）の読み取り。
 *
 * ページはこの形だけに依存する。必須項目（schema・run_id・datasets・models の key と label）が欠けたり
 * 型が違ったりしたら全体をエラーにし、数値の項目は欠けていれば null（画面では「—」）として扱う。
 * Why not zod: ページ専用に依存を 1 つ増やすほどの複雑さではなく、今後も項目が増えるので
 * 「無ければ null、あれば型を検査」という寛容な読み方を手で書いた方が意図がはっきりするため。
 */

/** このページが読める集計 JSON の版。score.py 側で形を変えたら上げる。 */
export const SUPPORTED_SCHEMA = 1;

/** モデル × データセット（または全体）1 グループ分の指標。CER は割合（0.123 = 12.3%）。 */
export interface GroupMetrics {
  n: number | null;
  failed: number | null;
  cerStrict: number | null;
  cerNorm: number | null;
  cerReading: number | null;
  rtf: number | null;
  /** クリップ単位の CER（正規化）の中央値。マイクロ平均が少数の長い誤りに引っ張られていないかを見る。 */
  cerNormMedian: number | null;
  /** クリップ単位の CER（正規化）が閾値（outlierThreshold、既定 100%）を超えたクリップ数（幻覚・繰り返し・参照の欠けなど）。 */
  outliers: number | null;
  /** 外れ値クリップを除いた CER（正規化）。感度を見るための参考値で、順位付けには使わない。 */
  cerNormExclOutliers: number | null;
}

export interface ModelResult {
  key: string;
  label: string;
  id: string | null;
  revision: string | null;
  /** その run で最初に読み込んだ秒（未キャッシュならダウンロードを含む）。JSON の first_load_s。 */
  loadFirstS: number | null;
  /**
   * 初回の読み込みがダウンロードを伴ったか（JSON の first_load_downloaded）。false はキャッシュ済みだったこと、
   * null は不明（項目が無い古い JSON を含む）を表す。
   */
  firstLoadDownloaded: boolean | null;
  /** ダウンロード済みのモデルをブラウザのキャッシュから読み込んだ秒。JSON の load_s。 */
  loadCachedS: number | null;
  /** 読み込み後、計測前に 1 回だけ流した文字起こしの秒（シェーダのコンパイル等）。 */
  warmupS: number | null;
  /** 予定件数に届かなかった（未実行の）クリップ数。 */
  missing: number | null;
  perDataset: Readonly<Record<string, GroupMetrics>>;
  overall: GroupMetrics | null;
  worstIds: Readonly<Record<string, readonly string[]>>;
}

export interface DatasetInfo {
  key: string;
  label: string;
  repo: string | null;
  revision: string | null;
  n: number | null;
  excludedOver30s: number | null;
  totalAudioS: number | null;
  description: string | null;
}

export interface BenchmarkResults {
  schema: number;
  runId: string;
  generatedAt: string | null;
  gitCommit: string | null;
  /** 集計した commit に未コミットの変更があったか。 */
  gitDirty: boolean;
  /** 外れ値とみなすクリップ単位 CER（正規化）の閾値（割合）。 */
  outlierThreshold: number;
  /** run 全体で外れ値として記録されたクリップの ID（JSON の outlier_ids）。項目が無い古い JSON では null。 */
  outlierIds: readonly string[] | null;
  /** 表示順を保った [項目名, 値] の組。 */
  environment: readonly (readonly [string, string])[];
  datasets: readonly DatasetInfo[];
  models: readonly ModelResult[];
  notes: readonly string[];
}

export type ParseResult = { ok: true; data: BenchmarkResults } | { ok: false; errors: string[] };

type JsonObject = Record<string, unknown>;

const isObject = (value: unknown): value is JsonObject =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** 外れ値の閾値が JSON に無いときの既定（クリップ単位 CER 100%）。eval/gemma4_eval/score.py の OUTLIER_CER_NORM と同じ値。 */
const DEFAULT_OUTLIER_THRESHOLD = 1;

/** GroupMetrics の各項目と、score.py（_page_group）が書く JSON の項目名の対応。 */
const GROUP_FIELDS = {
  n: 'n',
  failed: 'failed',
  cerStrict: 'cer_strict',
  cerNorm: 'cer_norm',
  cerReading: 'cer_reading',
  rtf: 'rtf',
  cerNormMedian: 'cer_norm_median',
  outliers: 'cer_norm_outliers',
  cerNormExclOutliers: 'cer_norm_excluding_outliers',
} as const satisfies Record<keyof GroupMetrics, string>;

class Reader {
  readonly errors: string[] = [];

  requiredString(object: JsonObject, key: string, path: string): string {
    const value = object[key];
    const isNonEmptyString = typeof value === 'string' && value.length > 0;
    if (isNonEmptyString) return value;
    this.errors.push(`${path}.${key} は空でない文字列である必要があります`);
    return '';
  }

  optionalString(object: JsonObject, key: string, path: string): string | null {
    const value = object[key];
    if (value === undefined || value === null) return null;
    if (typeof value === 'string') return value;
    this.errors.push(`${path}.${key} は文字列か null である必要があります`);
    return null;
  }

  /** 値は 0 以上の有限数か null（項目が無いときも null）。 */
  optionalNumber(object: JsonObject, key: string, path: string): number | null {
    const value = object[key];
    if (value === undefined || value === null) return null;
    const isValidNumber = typeof value === 'number' && Number.isFinite(value) && value >= 0;
    if (isValidNumber) return value;
    this.errors.push(`${path}.${key} は 0 以上の数値か null である必要があります`);
    return null;
  }

  /** 値は真偽値か null（項目が無いときも null）。 */
  optionalBoolean(object: JsonObject, key: string, path: string): boolean | null {
    const value = object[key];
    if (value === undefined || value === null) return null;
    if (typeof value === 'boolean') return value;
    this.errors.push(`${path}.${key} は真偽値か null である必要があります`);
    return null;
  }

  group(value: unknown, path: string): GroupMetrics | null {
    if (value === undefined || value === null) return null;
    if (!isObject(value)) {
      this.errors.push(`${path} はオブジェクトである必要があります`);
      return null;
    }
    const entries = Object.entries(GROUP_FIELDS).map(([name, key]) => [name, this.optionalNumber(value, key, path)]);
    return Object.fromEntries(entries) as unknown as GroupMetrics;
  }

  stringList(value: unknown, path: string): string[] {
    if (value === undefined || value === null) return [];
    const isStringArray = Array.isArray(value) && value.every((item) => typeof item === 'string');
    if (isStringArray) return value as string[];
    this.errors.push(`${path} は文字列の配列である必要があります`);
    return [];
  }
}

/** environment の値は文字列が基本だが、GPU 情報のようなオブジェクトは値を「 / 」でつなぐ。 */
function environmentValue(value: unknown): string | null {
  if (typeof value === 'string') return value || null;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (!isObject(value)) return null;
  const parts = Object.values(value).filter(
    (part): part is string | number => (typeof part === 'string' && part !== '') || typeof part === 'number',
  );
  return parts.length > 0 ? parts.join(' / ') : null;
}

function readDataset(reader: Reader, value: unknown, path: string): DatasetInfo | null {
  if (!isObject(value)) {
    reader.errors.push(`${path} はオブジェクトである必要があります`);
    return null;
  }
  return {
    key: reader.requiredString(value, 'key', path),
    label: reader.requiredString(value, 'label', path),
    repo: reader.optionalString(value, 'repo', path),
    revision: reader.optionalString(value, 'revision', path),
    n: reader.optionalNumber(value, 'n', path),
    excludedOver30s: reader.optionalNumber(value, 'excluded_over_30s', path),
    totalAudioS: reader.optionalNumber(value, 'total_audio_s', path),
    description: reader.optionalString(value, 'description', path),
  };
}

function readModel(reader: Reader, value: unknown, path: string): ModelResult | null {
  if (!isObject(value)) {
    reader.errors.push(`${path} はオブジェクトである必要があります`);
    return null;
  }
  const perDataset: Record<string, GroupMetrics> = {};
  const rawPerDataset = value['per_dataset'];
  const hasPerDataset = rawPerDataset !== undefined && rawPerDataset !== null;
  if (hasPerDataset && !isObject(rawPerDataset))
    reader.errors.push(`${path}.per_dataset はオブジェクトである必要があります`);
  if (isObject(rawPerDataset)) {
    for (const [dataset, metrics] of Object.entries(rawPerDataset)) {
      const group = reader.group(metrics, `${path}.per_dataset.${dataset}`);
      if (group !== null) perDataset[dataset] = group;
    }
  }
  const worstIds: Record<string, readonly string[]> = {};
  const rawWorst = value['worst_ids'];
  if (isObject(rawWorst)) {
    for (const [dataset, ids] of Object.entries(rawWorst)) {
      worstIds[dataset] = reader.stringList(ids, `${path}.worst_ids.${dataset}`);
    }
  }
  return {
    key: reader.requiredString(value, 'key', path),
    label: reader.requiredString(value, 'label', path),
    id: reader.optionalString(value, 'id', path),
    revision: reader.optionalString(value, 'revision', path),
    loadFirstS: reader.optionalNumber(value, 'first_load_s', path),
    firstLoadDownloaded: reader.optionalBoolean(value, 'first_load_downloaded', path),
    loadCachedS: reader.optionalNumber(value, 'load_s', path),
    warmupS: reader.optionalNumber(value, 'warmup_s', path),
    missing: reader.optionalNumber(value, 'missing', path),
    perDataset,
    overall: reader.group(value['overall'], `${path}.overall`),
    worstIds,
  };
}

function readList<T>(
  reader: Reader,
  root: JsonObject,
  key: string,
  read: (reader: Reader, value: unknown, path: string) => T | null,
): T[] {
  const value = root[key];
  if (!Array.isArray(value)) {
    reader.errors.push(`${key} は配列である必要があります`);
    return [];
  }
  return value.map((item, index) => read(reader, item, `${key}[${index}]`)).filter((item) => item !== null);
}

function duplicateKeys(items: readonly { key: string }[]): string[] {
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const { key } of items) {
    if (seen.has(key)) duplicates.add(key);
    seen.add(key);
  }
  return [...duplicates];
}

/** fetch した JSON（unknown）を検査して内部形式に変換する。不一致はすべて集めて返す。 */
export function parseResults(raw: unknown): ParseResult {
  if (!isObject(raw)) return { ok: false, errors: ['最上位がオブジェクトではありません'] };
  const reader = new Reader();

  const schema = raw['schema'];
  const isSupportedSchema = schema === SUPPORTED_SCHEMA;
  if (!isSupportedSchema) {
    return {
      ok: false,
      errors: [`schema ${JSON.stringify(schema)} には対応していません（対応: ${SUPPORTED_SCHEMA}）`],
    };
  }

  const runId = reader.requiredString(raw, 'run_id', '$');
  const generatedAt = reader.optionalString(raw, 'generated_at', '$');
  const gitCommit = reader.optionalString(raw, 'git_commit', '$');
  const gitDirty = raw['git_dirty'] === true;
  const outlierThreshold = reader.optionalNumber(raw, 'outlier_cer_norm_threshold', '$') ?? DEFAULT_OUTLIER_THRESHOLD;
  const datasets = readList(reader, raw, 'datasets', readDataset);
  const models = readList(reader, raw, 'models', readModel);
  const notes = reader.stringList(raw['notes'], 'notes');
  // 「項目が無い（古い JSON）」と「外れ値が 0 件」を区別するため、無ければ空配列ではなく null にする
  const hasOutlierIds = raw['outlier_ids'] !== undefined && raw['outlier_ids'] !== null;
  const outlierIds = hasOutlierIds ? reader.stringList(raw['outlier_ids'], 'outlier_ids') : null;

  const environment: (readonly [string, string])[] = [];
  const rawEnvironment = raw['environment'];
  if (isObject(rawEnvironment)) {
    for (const [name, value] of Object.entries(rawEnvironment)) {
      // gpu_adapter は gpu（文字列）の元になった内訳なので、gpu があれば二重に出さない
      const isDuplicateOfGpu = name === 'gpu_adapter' && typeof rawEnvironment['gpu'] === 'string';
      if (isDuplicateOfGpu) continue;
      const text = environmentValue(value);
      if (text !== null) environment.push([name, text]);
    }
  }

  for (const key of duplicateKeys(datasets)) reader.errors.push(`datasets の key「${key}」が重複しています`);
  for (const key of duplicateKeys(models)) reader.errors.push(`models の key「${key}」が重複しています`);

  if (reader.errors.length > 0) return { ok: false, errors: reader.errors };
  return {
    ok: true,
    data: {
      schema: SUPPORTED_SCHEMA,
      runId,
      generatedAt,
      gitCommit,
      gitDirty,
      outlierThreshold,
      outlierIds,
      environment,
      datasets,
      models,
      notes,
    },
  };
}
