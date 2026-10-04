import { decodeToMono16k, SAMPLING_RATE, splitIntoSegments } from '../audio.ts';
import { findModel } from '../models.ts';
import type { WorkerResponse } from '../protocol.ts';
import { TranscriptionClient } from './client.ts';
import {
  parseEvalParams,
  parseManifest,
  selectItems,
  type EvalResult,
  type EvalState,
  type ManifestItem,
} from './contract.ts';
import { countCachedModelFiles } from './model-cache.ts';

declare global {
  interface Window {
    /** Playwright（eval/gemma4_eval/run_browser.py）がポーリングする評価の進行状況。 */
    __eval: EvalState;
  }
}

const statusText = document.getElementById('status');
const setStatus = (text: string): void => {
  if (statusText !== null) statusText.textContent = text;
};

const errorText = (error: unknown): string => (error instanceof Error ? error.message : String(error));

// 引数の検証より先に公開する。検証で失敗しても Playwright が phase: 'error' と理由を読めるようにするため
const state: EvalState = {
  phase: 'loading-model',
  model: new URLSearchParams(location.search).get('model') ?? '',
  modelInfo: null,
  loadMs: null,
  warmupMs: null,
  currentId: null,
  cachedModelFiles: { before: null, after: null },
  total: 0,
  completed: 0,
  results: [],
  error: null,
};
// 名前は run_browser.py との契約。アプリ側の名前と衝突しないよう、あえて先頭に __ を付けている
// oxlint-disable-next-line no-underscore-dangle
window.__eval = state;

async function fetchManifest(url: URL): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url.href} の取得に失敗しました（HTTP ${response.status}）`);
  return response.json();
}

function spawnClient(): { client: TranscriptionClient; worker: Worker } {
  // アプリ（main.ts）と同じ worker.ts を同じ方法で読み込み、同じ推論経路を測る
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  const client = new TranscriptionClient((request, transfer) => worker.postMessage(request, transfer));
  worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => client.handleMessage(event.data));
  worker.addEventListener('error', (event) => {
    event.preventDefault();
    client.fail(event.message || 'Worker でエラーが発生しました');
  });
  worker.addEventListener('messageerror', () => client.fail('Worker からのメッセージを復元できませんでした'));
  return { client, worker };
}

/** 1 クリップを取得・デコードし、全区間を順に文字起こしする。失敗しても例外にせず error に記録して返す。 */
async function evaluateItem(
  client: TranscriptionClient,
  item: ManifestItem,
  audioUrl: URL,
  language: string,
): Promise<EvalResult> {
  const result: EvalResult = {
    id: item.id,
    dataset: item.dataset,
    reference: item.reference,
    hypothesis: '',
    durationS: 0,
    decodeMs: 0,
    inferMs: 0,
    segments: 0,
    error: null,
  };
  try {
    const decodeStartedAt = performance.now();
    const response = await fetch(audioUrl);
    if (!response.ok) throw new Error(`${audioUrl.href} の取得に失敗しました（HTTP ${response.status}）`);
    const audio = await decodeToMono16k(await response.blob());
    result.decodeMs = performance.now() - decodeStartedAt;
    result.durationS = audio.length / SAMPLING_RATE;

    const segments = splitIntoSegments(audio);
    result.segments = segments.length;
    // 空の仮説として CER に混ぜると「全部削除した誤り」と区別できないため、失敗として数える
    const isEmpty = segments.length === 0;
    if (isEmpty) throw new Error('音声が短すぎるか空です');

    for (const segment of segments) {
      const inferStartedAt = performance.now();
      try {
        // 区間の間に空白を挟まないのは、日本語の文中に余計な空白を作るだけで、CER はどの正規化でも空白を除去してから測るため
        result.hypothesis += await client.transcribe(segment.samples, language);
      } finally {
        result.inferMs += performance.now() - inferStartedAt;
      }
    }
  } catch (error) {
    result.error = errorText(error);
  }
  return result;
}

/**
 * 計測の前に、先頭のクリップを 1 回だけ計測せずに文字起こしする（結果は捨てる）。
 * 最初の推論には WebGPU のシェーダのコンパイルや AudioContext・デコーダの初期化が乗り、先頭クリップだけが遅くなるため。
 * Why not 1 秒の無音: 無音では生成がすぐ終わる（または無音を埋める文を延々と生成する）ため実際の推論と経路・長さが違い、
 * fetch と decodeAudioData の初期化も温まらない。先頭クリップならタイムアウトやクラッシュも本番と同じ ID で扱える
 */
async function warmUp(
  client: TranscriptionClient,
  item: ManifestItem,
  audioUrl: URL,
  language: string,
): Promise<number> {
  const startedAt = performance.now();
  // クリップ単位の失敗（音声が空など）は本番の計測で同じクリップの失敗として記録されるので、ここでは無視する
  const result = await evaluateItem(client, item, audioUrl, language);
  const hasWorkerFailed = client.failure !== null;
  if (hasWorkerFailed) {
    // 計測ループで Worker が落ちたときと同じく、そのクリップの失敗として結果に残してから止める。
    // 残さないと run_browser は開き直すたびに同じクリップのウォームアップで落ち、先へ進めないまま諦めることになる
    state.results.push(result);
    state.completed++;
    throw new Error(`ウォームアップ中に Worker が停止しました: ${client.failure}`);
  }
  return performance.now() - startedAt;
}

async function run(): Promise<void> {
  const params = parseEvalParams(location.search);
  state.model = params.model;
  const { key, label, id, revision, dtype } = findModel(params.model);
  state.modelInfo = { key, label, id, revision, dtype };

  setStatus('manifest を読み込み中…');
  const manifestUrl = new URL(params.manifest, location.href);
  const manifest = parseManifest(await fetchManifest(manifestUrl));
  const items = selectItems(manifest.items, params.offset, params.limit);
  state.total = items.length;

  // モデルの数 GB のダウンロードを始める前に確かめる。WebGPU 無しで読み込むと、原因の分かりにくい ORT のエラーになるため
  const adapter = 'gpu' in navigator ? await navigator.gpu.requestAdapter() : null;
  const hasWebGpu = adapter !== null;
  if (!hasWebGpu) throw new Error('WebGPU が使えません');

  // 安全でないコンテキストなどでは caches が無い。そのときは null（ダウンロードの有無は不明）として続ける
  const cacheStorage = 'caches' in globalThis ? caches : undefined;
  state.cachedModelFiles.before = await countCachedModelFiles(cacheStorage, id, revision);

  const { client, worker } = spawnClient();
  client.onLoadProgress = (progress) => setStatus(`${params.model} をダウンロード中… ${progress.toFixed(1)}%`);
  try {
    setStatus(`${params.model} を読み込み中…`);
    const loadMs = await client.load(params.model);
    // loadMs より先に入れる。run_browser が loadMs を見た時点で after も揃っているようにするため
    state.cachedModelFiles.after = await countCachedModelFiles(cacheStorage, id, revision);
    state.loadMs = loadMs;

    // wav の path は manifest.json からの相対なので、manifest の URL を基準に解決する
    const audioUrlOf = (item: ManifestItem): URL => new URL(item.path, manifestUrl);
    // limit=0（run_browser のダウンロード専用セッション）ではクリップが無いので温めない
    const [first] = items;
    if (first !== undefined) {
      state.phase = 'warming-up';
      state.currentId = first.id;
      setStatus(`${params.model}: ウォームアップ中（${first.id}、計測しない）`);
      state.warmupMs = await warmUp(client, first, audioUrlOf(first), params.language);
      state.currentId = null;
    }
    state.phase = 'running';

    for (const [index, item] of items.entries()) {
      setStatus(`${params.model}: ${index + 1} / ${items.length}（${item.id}）`);
      state.currentId = item.id;
      const result = await evaluateItem(client, item, audioUrlOf(item), params.language);
      // completed より先に push する。completed を見て results を読む側が、未反映の結果を読み落とさないため
      state.results.push(result);
      state.completed++;
      state.currentId = null;
      // Worker が落ちたあとの残りは全件同じ理由で失敗するだけなので、個別失敗として続けず評価全体を止める
      const hasWorkerFailed = client.failure !== null;
      if (hasWorkerFailed) throw new Error(`Worker が停止しました: ${client.failure}`);
    }
  } finally {
    // GPU メモリを手放す。結果は window.__eval に残る
    worker.terminate();
  }

  state.phase = 'done';
  const failed = state.results.filter((result) => result.error !== null).length;
  setStatus(`完了: ${state.completed} 件（失敗 ${failed} 件）`);
}

run().catch((error: unknown) => {
  console.error(error);
  state.phase = 'error';
  state.error = errorText(error);
  setStatus(`エラー: ${state.error}`);
});
