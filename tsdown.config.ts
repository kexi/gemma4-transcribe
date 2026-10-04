import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

import { defineConfig } from 'tsdown';

// Transformers.js が依存している onnxruntime-web の dist を、Transformers.js 自身の位置から解決する。
// 直接依存に足さないのは、Transformers.js 側の ORT 更新と版がずれて JS と WASM が食い違うのを避けるため
const requireFromTransformers = createRequire(createRequire(import.meta.url).resolve('@huggingface/transformers'));
const ortDist = path.dirname(requireFromTransformers.resolve('onnxruntime-web'));
const ortVersion: string = JSON.parse(readFileSync(path.join(ortDist, '..', 'package.json'), 'utf8')).version;
// Transformers.js は ORT の WASM を URL をキーに Cache API へ永続化し再検証しないため、版をパスに含めて更新時に別キーにする
const ortDir = `ort/${ortVersion}`;

export default defineConfig({
  entry: { main: 'src/main.ts', worker: 'src/worker.ts' },
  platform: 'browser',
  format: 'esm',
  target: 'es2024',
  outDir: 'dist',
  // GitHub Pages に置く静的サイトなので、ライブラリ向けの既定（dependencies を外部化）を外して全部同梱する
  deps: { alwaysBundle: [/.*/], onlyBundle: false },
  // main.ts は new URL('./worker.js', import.meta.url) で worker を参照するため、ファイル名にハッシュを付けない
  hash: false,
  dts: false,
  sourcemap: true,
  minify: true,
  // worker.ts の wasmPaths とコピー先を同じ値から作り、パスの食い違いで 404 にならないようにする
  define: { BUILD_ORT_DIR: JSON.stringify(ortDir) },
  copy: [
    { from: 'public/*', flatten: true },
    // jsDelivr から実行時に取らず、lockfile で検証済みの ORT ランタイムを同一オリジンから配信する（worker.ts の wasmPaths と対応）
    {
      from: [
        path.join(ortDist, 'ort-wasm-simd-threaded.asyncify.mjs'),
        path.join(ortDist, 'ort-wasm-simd-threaded.asyncify.wasm'),
      ],
      to: `dist/${ortDir}`,
      flatten: true,
    },
  ],
});
