import { defineConfig } from 'tsdown';

import { benchmarkResultsCopy, ortRuntimeCopy, sharedConfig } from './tsdown.shared.ts';

const outDir = 'dist';

export default defineConfig({
  ...sharedConfig,
  // benchmarks は公開ベンチマークページ（public/benchmarks.html）。評価ページ（src/eval/）とは違い本番に載せる
  entry: { main: 'src/main.ts', worker: 'src/worker.ts', benchmarks: 'src/benchmarks.ts' },
  outDir,
  copy: [{ from: 'public/*', flatten: true }, ortRuntimeCopy(outDir), ...benchmarkResultsCopy(outDir)],
});
