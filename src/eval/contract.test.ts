import { describe, expect, it } from 'vitest';

import { DEFAULT_LANGUAGE, DEFAULT_MANIFEST_URL, parseEvalParams, parseManifest, selectItems } from './contract.ts';

describe('parseEvalParams', () => {
  it('model だけ指定すれば、manifest・言語・範囲は契約どおりの既定になる', () => {
    expect(parseEvalParams('?model=e2b')).toEqual({
      model: 'e2b',
      manifest: DEFAULT_MANIFEST_URL,
      limit: null,
      offset: 0,
      language: DEFAULT_LANGUAGE,
    });
  });

  it('URL のすべての引数を読む', () => {
    expect(parseEvalParams('?model=e4b-qat&manifest=/data/other.json&limit=10&offset=20&language=English')).toEqual({
      model: 'e4b-qat',
      manifest: '/data/other.json',
      limit: 10,
      offset: 20,
      language: 'English',
    });
  });

  it('空の limit / offset は未指定として扱う（just の limit="" がそのまま渡っても全件になる）', () => {
    const params = parseEvalParams('?model=e2b&limit=&offset=');
    expect(params.limit).toBeNull();
    expect(params.offset).toBe(0);
  });

  it('未知のモデルは既定モデルに読み替えず失敗する（別モデルの結果を取り違えない）', () => {
    expect(() => parseEvalParams('?model=e3b')).toThrow(/e3b/);
    expect(() => parseEvalParams('')).toThrow(/model/);
  });

  it('未知の言語は失敗する', () => {
    expect(() => parseEvalParams('?model=e2b&language=Klingon')).toThrow(/Klingon/);
  });

  it('負数・小数・数字以外の limit / offset は失敗する', () => {
    expect(() => parseEvalParams('?model=e2b&limit=-1')).toThrow(/limit/);
    expect(() => parseEvalParams('?model=e2b&limit=1.5')).toThrow(/limit/);
    expect(() => parseEvalParams('?model=e2b&offset=abc')).toThrow(/offset/);
  });
});

describe('parseManifest', () => {
  const item = { id: 'jsut-000001', dataset: 'jsut', path: 'jsut/jsut-000001.wav', duration_s: 3.2, reference: 'あ' };

  it('契約どおりの items を返す（余分なキーは捨てる）', () => {
    const manifest = parseManifest({ per_dataset: 1, items: [{ ...item, extra: true }] });
    expect(manifest.items).toEqual([item]);
  });

  it('items が無ければ失敗する', () => {
    expect(() => parseManifest({})).toThrow(/items/);
    expect(() => parseManifest(null)).toThrow(/items/);
  });

  it('型が違うフィールドを持つ item があれば、その位置を示して失敗する', () => {
    expect(() => parseManifest({ items: [item, { ...item, duration_s: '3.2' }] })).toThrow(/items\[1\]/);
  });
});

describe('selectItems', () => {
  const items = [0, 1, 2, 3, 4];

  it('offset から limit 件を取り出す', () => {
    expect(selectItems(items, 1, 2)).toEqual([1, 2]);
  });

  it('limit が null なら offset 以降をすべて取り出す', () => {
    expect(selectItems(items, 3, null)).toEqual([3, 4]);
  });

  it('範囲が末尾を越えても、ある分だけを返す', () => {
    expect(selectItems(items, 4, 10)).toEqual([4]);
    expect(selectItems(items, 10, null)).toEqual([]);
  });
});
