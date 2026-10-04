import { defineConfig } from 'tsdown';

import { ortRuntimeCopy, sharedConfig } from './tsdown.shared.ts';

// 評価ページは公開しないので、GitHub Pages に載る dist/ とは別の出力先に分ける。
// tsdown.config.ts に entry を足さないのは、評価用のコードと eval.html が本番の dist/ に混ざるのを避けるため
const outDir = 'dist-eval';

export default defineConfig({
  ...sharedConfig,
  // worker はアプリと同じ src/worker.ts。runner.ts も new URL('./worker.js', import.meta.url) で同じディレクトリから読む
  entry: { eval: 'src/eval/runner.ts', worker: 'src/worker.ts' },
  outDir,
  copy: [{ from: 'eval/web/eval.html', flatten: true }, ortRuntimeCopy(outDir)],
});
