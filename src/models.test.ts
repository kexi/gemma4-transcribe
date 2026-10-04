import { describe, expect, it } from 'vitest';

import { findModel, MODELS } from './models.ts';

describe('MODELS', () => {
  it('キーが重複しない', () => {
    const keys = MODELS.map((model) => model.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it('すべて 40 桁のコミット SHA に固定されている（ブランチ名で取りに行かない）', () => {
    for (const model of MODELS) expect(model.revision).toMatch(/^[0-9a-f]{40}$/);
  });

  it('すべて音声エンコーダを持つ Gemma 4 E2B / E4B である', () => {
    for (const model of MODELS) expect(model.id).toMatch(/^onnx-community\/gemma-4-E[24]B-it-/);
  });

  it('セッション別 dtype を持つモデルは音声エンコーダの dtype を指定している', () => {
    for (const { dtype } of MODELS) {
      if (typeof dtype === 'string') continue;
      expect(dtype).toHaveProperty('audio_encoder');
    }
  });
});

describe('findModel', () => {
  it('キーに一致するモデルを返す', () => {
    expect(findModel('e4b').id).toBe('onnx-community/gemma-4-E4B-it-ONNX');
  });

  it.each([null, undefined, '', 'unknown'])('不明なキー %s は既定（先頭）のモデルにする', (key) => {
    expect(findModel(key)).toBe(MODELS[0]);
  });
});
