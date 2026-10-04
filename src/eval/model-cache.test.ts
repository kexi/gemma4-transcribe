import { describe, expect, it } from 'vitest';

import { countCachedModelFiles, modelFileKeyPart, type CacheStorageLike } from './model-cache.ts';

const ID = 'onnx-community/gemma-4-E2B-it-ONNX';
const REVISION = 'a'.repeat(40);
const fileUrl = (file: string, revision = REVISION, id = ID): string =>
  `https://huggingface.co/${id}/resolve/${revision}/${file}`;

function fakeStorage(caches: Record<string, string[]>): CacheStorageLike {
  return {
    keys: async () => Object.keys(caches),
    open: async (name) => ({ keys: async () => (caches[name] ?? []).map((url) => ({ url })) }),
  };
}

describe('modelFileKeyPart', () => {
  it('Transformers.js が保存するキー（<id>/resolve/<revision>/）に一致する部分を返す', () => {
    expect(fileUrl('config.json')).toContain(modelFileKeyPart(ID, REVISION));
  });
});

describe('countCachedModelFiles', () => {
  it('どのキャッシュにあっても、このモデル・この revision のファイルだけを数える', async () => {
    const storage = fakeStorage({
      'transformers-cache': [
        fileUrl('config.json'),
        fileUrl('onnx/decoder_model_merged_q4f16.onnx'),
        // 別の revision と別のモデルは数えない（保存済みと誤って判断しないため）
        fileUrl('config.json', 'b'.repeat(40)),
        fileUrl('config.json', REVISION, 'onnx-community/gemma-4-E4B-it-ONNX'),
        // 同一オリジンの ORT の WASM はモデルのダウンロードではない
        'http://127.0.0.1:8765/ort/1.23.0/ort-wasm-simd-threaded.asyncify.wasm',
      ],
      'renamed-cache': [fileUrl('tokenizer.json')],
    });
    await expect(countCachedModelFiles(storage, ID, REVISION)).resolves.toBe(3);
  });

  it('キャッシュが空なら 0（null ではない。ダウンロード後に増えたかを比べられる）', async () => {
    await expect(countCachedModelFiles(fakeStorage({}), ID, REVISION)).resolves.toBe(0);
  });

  it('Cache API が無い・読めないときは null を返し、評価を止めない', async () => {
    await expect(countCachedModelFiles(undefined, ID, REVISION)).resolves.toBeNull();
    const denied: CacheStorageLike = {
      keys: async () => {
        throw new Error('SecurityError');
      },
      open: async () => ({ keys: async () => [] }),
    };
    await expect(countCachedModelFiles(denied, ID, REVISION)).resolves.toBeNull();
  });
});
