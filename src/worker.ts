import {
  AutoProcessor,
  env,
  Gemma4ForConditionalGeneration,
  InterruptableStoppingCriteria,
  TextStreamer,
  type ProgressInfo,
} from '@huggingface/transformers';

import { buildTranscriptionPrompt } from './prompt.ts';
import type { WorkerRequest, WorkerResponse } from './protocol.ts';

const MODEL_ID = 'onnx-community/gemma-4-E2B-it-ONNX';
/**
 * Hugging Face 上のモデルリポジトリのコミット SHA。
 * main を追うと第三者のリポジトリ更新がそのまま全訪問者に届くため、更新はこの値を変えるコミットとしてレビューする
 */
const MODEL_REVISION = '9f4bef82ea6e296bc69f8a2f5939f73af81b07a6';
/** 30 秒の発話を書き起こすには十分で、暴走時に打ち切れる上限。 */
const MAX_NEW_TOKENS = 512;

// Transformers.js の既定は jsDelivr から ORT ランタイムを取るが、tsdown が dist/ に同梱したものを同一オリジンから使う
const onnxWasm = env.backends.onnx.wasm;
if (onnxWasm !== undefined) {
  onnxWasm.wasmPaths = {
    mjs: new URL(`./${BUILD_ORT_DIR}/ort-wasm-simd-threaded.asyncify.mjs`, import.meta.url).href,
    wasm: new URL(`./${BUILD_ORT_DIR}/ort-wasm-simd-threaded.asyncify.wasm`, import.meta.url).href,
  };
}

type Processor = Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
type Model = Awaited<ReturnType<typeof Gemma4ForConditionalGeneration.from_pretrained>>;

let loaded: Promise<{ processor: Processor; model: Model }> | undefined;
/** 読み込みの試行番号。失敗した試行の取り残されたダウンロードが進捗を送り続けるのを止めるために使う。 */
let currentLoadAttempt = 0;
const stopping = new InterruptableStoppingCriteria();

const post = (message: WorkerResponse): void => {
  // DOM lib では self が Window 型になるため lint が targetOrigin を要求するが、Worker の postMessage にその引数は無い
  // oxlint-disable-next-line unicorn/require-post-message-target-origin
  self.postMessage(message);
};

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

async function loadModel(attempt: number): Promise<{ processor: Processor; model: Model }> {
  const startedAt = performance.now();
  const onProgress = (info: ProgressInfo): void => {
    const isStaleAttempt = attempt !== currentLoadAttempt;
    const isTotalProgress = info.status === 'progress_total';
    if (isStaleAttempt || !isTotalProgress) return;
    post({ type: 'load-progress', progress: info.progress, loadedBytes: info.loaded, totalBytes: info.total });
  };

  // Promise.all ではなく allSettled にするのは、片方だけ成功したときにモデル側の WebGPU セッションを解放してから失敗させるため
  const [processor, model] = await Promise.allSettled([
    AutoProcessor.from_pretrained(MODEL_ID, { revision: MODEL_REVISION }),
    Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
      revision: MODEL_REVISION,
      // q4f16 が WebGPU で最小（約 3.4GB）。q4 は fp32 演算になり VRAM と転送量が増えるため採らない
      dtype: 'q4f16',
      device: 'webgpu',
      progress_callback: onProgress,
    }),
  ]);
  if (processor.status === 'rejected' || model.status === 'rejected') {
    if (model.status === 'fulfilled') await model.value.dispose();
    throw processor.status === 'rejected' ? processor.reason : (model as PromiseRejectedResult).reason;
  }
  post({ type: 'loaded', elapsedMs: performance.now() - startedAt });
  return { processor: processor.value, model: model.value };
}

async function transcribe(id: number, samples: Float32Array, language: string): Promise<void> {
  if (loaded === undefined) throw new Error('モデルが読み込まれていません');
  const { processor, model } = await loaded;

  const messages = [
    {
      role: 'user',
      // モデルカードの推奨どおり、音声をテキストより前に置く
      content: [{ type: 'audio' }, { type: 'text', text: buildTranscriptionPrompt(language) }],
    },
  ];
  // enable_thinking は chat template へそのまま渡る追加引数で、Transformers.js の型には無いため変数経由で渡す
  const templateOptions = { tokenize: false, add_generation_prompt: true, enable_thinking: false } as const;
  const prompt = processor.apply_chat_template(messages, templateOptions);
  if (typeof prompt !== 'string') throw new Error('chat template の展開結果が文字列ではありません');
  const tokenizer = processor.tokenizer;
  if (tokenizer === undefined) throw new Error('processor に tokenizer がありません');
  const inputs = await processor(prompt, null, samples, { add_special_tokens: false });

  await model.generate({
    ...inputs,
    max_new_tokens: MAX_NEW_TOKENS,
    do_sample: false,
    stopping_criteria: stopping,
    streamer: new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => post({ type: 'token', id, text }),
    }),
  });
  post({ type: 'done', id, interrupted: stopping.interrupted });
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  switch (request.type) {
    case 'load': {
      if (loaded !== undefined) return;
      const attempt = ++currentLoadAttempt;
      const attemptPromise = loadModel(attempt);
      loaded = attemptPromise;
      attemptPromise.catch((error: unknown) => {
        // 失敗した Promise を握り続けると再試行できないので捨て、取り残されたダウンロードの進捗も止める
        if (loaded === attemptPromise) loaded = undefined;
        currentLoadAttempt++;
        post({ type: 'load-error', message: errorMessage(error) });
      });
      return;
    }
    case 'transcribe': {
      // transcribe() 内の await より前（メッセージ受信時点）で解除しておく。
      // 前処理中に届いた interrupt を、あとから reset() で消してしまわないため
      stopping.reset();
      transcribe(request.id, request.samples, request.language).catch((error: unknown) => {
        post({ type: 'transcribe-error', id: request.id, message: errorMessage(error) });
      });
      return;
    }
    case 'interrupt': {
      stopping.interrupt();
      return;
    }
  }
});
