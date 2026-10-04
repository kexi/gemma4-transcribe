import { globSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

import type { CopyEntry, UserConfig } from 'tsdown';

// Transformers.js が依存している onnxruntime-web の dist を、Transformers.js 自身の位置から解決する。
// 直接依存に足さないのは、Transformers.js 側の ORT 更新と版がずれて JS と WASM が食い違うのを避けるため
const requireFromTransformers = createRequire(createRequire(import.meta.url).resolve('@huggingface/transformers'));
const ortDist = path.dirname(requireFromTransformers.resolve('onnxruntime-web'));
const ortVersion: string = JSON.parse(readFileSync(path.join(ortDist, '..', 'package.json'), 'utf8')).version;
// Transformers.js は ORT の WASM を URL をキーに Cache API へ永続化し再検証しないため、版をパスに含めて更新時に別キーにする
const ortDir = `ort/${ortVersion}`;

/**
 * アプリ本体（tsdown.config.ts）と評価ページ（tsdown.eval.config.ts）に共通のビルド設定。
 * 評価ページはアプリと同じ worker.ts を同じ条件でビルドしないと「アプリと同じ推論経路」を測れないため、
 * 設定を複製せずここから両方に配る
 */
export const sharedConfig = {
  platform: 'browser',
  format: 'esm',
  target: 'es2024',
  // GitHub Pages に置く静的サイトなので、ライブラリ向けの既定（dependencies を外部化）を外して全部同梱する
  deps: { alwaysBundle: [/.*/], onlyBundle: false },
  // 呼び出し側は new URL('./worker.js', import.meta.url) で worker を参照するため、ファイル名にハッシュを付けない
  hash: false,
  dts: false,
  sourcemap: true,
  minify: true,
  // worker.ts の wasmPaths とコピー先を同じ値から作り、パスの食い違いで 404 にならないようにする
  define: { BUILD_ORT_DIR: JSON.stringify(ortDir) },
} satisfies UserConfig;

/**
 * jsDelivr から実行時に取らず、lockfile で検証済みの ORT ランタイムを同一オリジンから配信する（worker.ts の wasmPaths と対応）。
 * `outDir` は各設定の出力先で、worker.js と同じディレクトリを基準に `BUILD_ORT_DIR` へ置く。
 */
export function ortRuntimeCopy(outDir: string): CopyEntry {
  return {
    from: [
      path.join(ortDist, 'ort-wasm-simd-threaded.asyncify.mjs'),
      path.join(ortDist, 'ort-wasm-simd-threaded.asyncify.wasm'),
    ],
    to: `${outDir}/${ortDir}`,
    flatten: true,
  };
}

/** 評価ハーネス（just eval-score）が書く集計 JSON。数値と ID だけなのでそのまま公開してよい。 */
const BENCHMARK_RESULTS_GLOB = 'eval/results/*.json';

/**
 * 集計 JSON を `<outDir>/benchmarks/` にコピーする（benchmarks.html が ./benchmarks/latest.json を読む）。
 * まだ結果が 1 つも無いときは何もコピーせず、ページ側が「まだ結果がありません」を出す。
 * Why not glob をそのまま copy に渡す: 一致が 0 件のときの扱いをツールの挙動に任せず、結果が無くてもビルドが必ず通るようにするため。
 * Why not 公開用の複製を public/ に置く: 同じ数値がリポジトリに 2 か所あると、片方だけ更新される食い違いが起きるため
 */
export function benchmarkResultsCopy(outDir: string): CopyEntry[] {
  const files = globSync(BENCHMARK_RESULTS_GLOB);
  if (files.length === 0) return [];
  return [{ from: files, to: `${outDir}/benchmarks`, flatten: true }];
}
