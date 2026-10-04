import { describe, expect, it } from 'vitest';

import fixture from './fixture.json' with { type: 'json' };
import { parseResults, type BenchmarkResults } from './schema.ts';

/** フィクスチャを壊さないよう、テストごとに深いコピーを作って書き換える。 */
const clone = (): Record<string, unknown> => structuredClone(fixture) as Record<string, unknown>;

/** 失敗したときのエラー一覧。成功したら空配列（条件付きの expect を避けるため、結果を値にしてから検査する）。 */
const errorsOf = (raw: unknown): string[] => {
  const result = parseResults(raw);
  return result.ok ? [] : result.errors;
};

const parseOk = (raw: unknown): BenchmarkResults => {
  const result = parseResults(raw);
  if (!result.ok) throw new Error(result.errors.join('\n'));
  return result.data;
};

describe('parseResults', () => {
  it('仕様どおりの集計 JSON（フィクスチャ）を読める', () => {
    const data = parseOk(fixture);
    expect(data.runId).toBe('20261004T120000Z');
    expect(data.datasets.map((dataset) => dataset.key)).toEqual(['jsut', 'reazon', 'cv']);
    expect(data.models.map((model) => model.key)).toEqual(['e2b', 'e2b-qat', 'e4b', 'e4b-qat']);
    expect(data.models[0]?.overall?.cerNorm).toBe(0.205);
    expect(data.models[0]?.perDataset['reazon']?.outliers).toBe(4);
  });

  it('GPU のようなオブジェクトの環境情報は、空でない値を「 / 」でつないだ文字列にする', () => {
    const raw = { ...clone(), environment: { gpu: { vendor: 'apple', architecture: 'metal-3', device: '' } } };
    expect(parseOk(raw).environment).toEqual([['gpu', 'apple / metal-3']]);
  });

  it('gpu_adapter は gpu（文字列）の内訳なので、gpu があれば二重に出さない', () => {
    const names = parseOk(fixture).environment.map(([name]) => name);
    expect(names).toContain('gpu');
    expect(names).not.toContain('gpu_adapter');
  });

  it('任意の数値項目が無い・null のときは null にして全体はエラーにしない', () => {
    const data = parseOk(fixture);
    const qat = data.models.find((model) => model.key === 'e2b-qat');
    expect(qat?.warmupS).toBeNull();
    expect(qat?.perDataset['jsut']?.outliers).toBeNull();
    const unmeasured = data.models.find((model) => model.key === 'e4b-qat');
    expect(unmeasured?.overall).toBeNull();
    expect(unmeasured?.loadFirstS).toBeNull();
    expect(unmeasured?.perDataset).toEqual({});
  });

  it('load_s はキャッシュからの読み込み秒、first_load_s は初回の読み込み秒として読む（score.py の定義）', () => {
    const e2b = parseOk(fixture).models[0];
    expect(e2b?.loadCachedS).toBe(6.8);
    expect(e2b?.loadFirstS).toBe(132.5);
  });

  it('git_dirty と外れ値の閾値を読み、閾値が無ければ既定の 100%（1）にする', () => {
    const data = parseOk({ ...clone(), git_dirty: true });
    expect(data.gitDirty).toBe(true);
    expect(data.outlierThreshold).toBe(1);
    const withoutThreshold = clone();
    delete withoutThreshold['outlier_cer_norm_threshold'];
    expect(parseOk(withoutThreshold).outlierThreshold).toBe(1);
    expect(parseOk({ ...clone(), outlier_cer_norm_threshold: 0.8 }).outlierThreshold).toBe(0.8);
  });

  it('対応していない schema の版は中身を読まずにエラーにする', () => {
    const raw = { ...clone(), schema: 2 };
    expect(errorsOf(raw)[0]).toContain('schema 2');
  });

  it.each([null, [], 'text', 1])('最上位がオブジェクトでない値 %j はエラーにする', (raw) => {
    expect(parseResults(raw).ok).toBe(false);
  });

  it('必須項目（models の key）が欠けるとその場所を示してエラーにする', () => {
    const raw = clone();
    const models = raw['models'] as Record<string, unknown>[];
    delete models[1]?.['key'];
    expect(errorsOf(raw)).toContain('models[1].key は空でない文字列である必要があります');
  });

  it('数値項目に文字列や負の値が入っていたら、黙って「—」にせずエラーにする', () => {
    const raw = clone();
    const models = raw['models'] as { overall: Record<string, unknown> }[];
    const first = models[0];
    if (first === undefined) throw new Error('fixture に models がありません');
    first.overall['cer_norm'] = '20.5';
    first.overall['rtf'] = -1;
    const errors = errorsOf(raw);
    expect(errors).toContain('models[0].overall.cer_norm は 0 以上の数値か null である必要があります');
    expect(errors).toContain('models[0].overall.rtf は 0 以上の数値か null である必要があります');
  });

  it('datasets / models が配列でなければエラーにする', () => {
    expect(errorsOf({ ...clone(), datasets: {}, models: null })).toEqual([
      'datasets は配列である必要があります',
      'models は配列である必要があります',
    ]);
  });

  it('同じ key のモデルが 2 つあると色と行が食い違うのでエラーにする', () => {
    const raw = clone();
    const models = raw['models'] as Record<string, unknown>[];
    raw['models'] = [...models, models[0]];
    expect(errorsOf(raw)).toContain('models の key「e2b」が重複しています');
  });

  it('未実行のクリップ数（missing）を読む', () => {
    const unmeasured = parseOk(fixture).models.find((model) => model.key === 'e4b-qat');
    expect(unmeasured?.missing).toBe(300);
  });

  it('first_load_downloaded（真偽値・null）と run 全体の outlier_ids を読む', () => {
    const raw = clone();
    const models = raw['models'] as Record<string, unknown>[];
    const [first, second] = models;
    if (first === undefined || second === undefined) throw new Error('fixture に models がありません');
    first['first_load_downloaded'] = true;
    second['first_load_downloaded'] = false;
    raw['outlier_ids'] = ['reazon-003315', 'reazon-003858'];
    const data = parseOk(raw);
    expect(data.models.map((model) => model.firstLoadDownloaded)).toEqual([true, false, null, null]);
    expect(data.outlierIds).toEqual(['reazon-003315', 'reazon-003858']);
  });

  it('first_load_downloaded・outlier_ids の無い古い形の JSON も読み、どちらも不明（null）にする', () => {
    const data = parseOk(fixture);
    expect(data.models.every((model) => model.firstLoadDownloaded === null)).toBe(true);
    expect(data.outlierIds).toBeNull();
    expect(parseOk({ ...clone(), outlier_ids: [] }).outlierIds).toEqual([]);
  });

  it('first_load_downloaded・outlier_ids の型が違えばエラーにする', () => {
    const raw = clone();
    const models = raw['models'] as Record<string, unknown>[];
    const [first] = models;
    if (first === undefined) throw new Error('fixture に models がありません');
    first['first_load_downloaded'] = 'yes';
    raw['outlier_ids'] = [1, 2];
    const errors = errorsOf(raw);
    expect(errors).toContain('models[0].first_load_downloaded は真偽値か null である必要があります');
    expect(errors).toContain('outlier_ids は文字列の配列である必要があります');
  });
});
