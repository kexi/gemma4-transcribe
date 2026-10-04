import { decodeToMono16k, formatTimestamp, SAMPLING_RATE, splitIntoSegments, type AudioSegment } from './audio.ts';
import { LANGUAGES } from './prompt.ts';
import type { WorkerRequest, WorkerResponse } from './protocol.ts';

/**
 * 画面全体の状態。ボタンの有効・無効はすべてこの値から決める。
 * - acquiring: getUserMedia の許可待ち（この間の再クリックで録音が二重に始まらないよう、全入力を止める）
 * - finishing: recorder.stop() から 'stop' イベントで文字起こしが始まるまでの隙間
 */
type AppState = 'unloaded' | 'loading' | 'idle' | 'acquiring' | 'recording' | 'finishing' | 'transcribing';

function mustGet<T extends HTMLElement>(id: string, type: new () => T): T {
  const element = document.getElementById(id);
  if (!(element instanceof type)) throw new Error(`#${id} が見つかりません`);
  return element;
}

const loadButton = mustGet('load', HTMLButtonElement);
const progressBar = mustGet('progress', HTMLProgressElement);
const statusText = mustGet('status', HTMLParagraphElement);
const languageSelect = mustGet('language', HTMLSelectElement);
const recordButton = mustGet('record', HTMLButtonElement);
const fileInput = mustGet('file', HTMLInputElement);
const stopButton = mustGet('stop', HTMLButtonElement);
const player = mustGet('player', HTMLAudioElement);
const output = mustGet('output', HTMLDivElement);

for (const { name, label } of LANGUAGES) languageSelect.add(new Option(label, name));

/** 区間 ID ごとの、完了待ちと書き込み先。 */
const pending = new Map<
  number,
  { target: HTMLElement; resolve: (interrupted: boolean) => void; reject: (error: Error) => void }
>();
let nextId = 0;
let state: AppState = 'unloaded';
/** 文字起こし後に戻る先を決めるため、Worker 内にモデルが載っているかを覚えておく。 */
let isModelLoaded = false;
/** Worker が落ちた理由。モデルが失われたことを、次の操作のときに利用者へ伝えるために残す。 */
let workerFailure: string | undefined;
let isStopRequested = false;
let recorder: MediaRecorder | undefined;
let playerUrl: string | undefined;

const setStatus = (text: string): void => {
  statusText.textContent = text;
};

const errorText = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const formatMegabytes = (bytes: number): string => `${(bytes / 1024 / 1024).toFixed(0)}MB`;

/** 一連の操作が終わったときの戻り先。Worker が落ちてモデルを失っていれば読み込み前に戻す。 */
function settle(): void {
  setState(isModelLoaded ? 'idle' : 'unloaded');
}

function setState(next: AppState): void {
  state = next;
  const isIdle = state === 'idle';
  const isRecording = state === 'recording';
  loadButton.disabled = state !== 'unloaded';
  progressBar.hidden = state !== 'loading';
  recordButton.disabled = !(isIdle || isRecording);
  recordButton.textContent = isRecording ? '■ 録音停止' : '● 録音開始';
  recordButton.classList.toggle('recording', isRecording);
  fileInput.disabled = !isIdle;
  languageSelect.disabled = !isIdle;
  stopButton.disabled = state !== 'transcribing' || isStopRequested;
}

function handleWorkerMessage(event: MessageEvent<WorkerResponse>): void {
  // terminate() 前にキューへ積まれていた、捨てた Worker からのメッセージは無視する
  const isFromCurrentWorker = event.currentTarget === worker;
  if (!isFromCurrentWorker) return;
  const message = event.data;
  switch (message.type) {
    case 'load-progress': {
      const isLoading = state === 'loading';
      if (!isLoading) return;
      progressBar.value = message.progress;
      setStatus(
        `ダウンロード中… ${message.progress.toFixed(1)}%（${formatMegabytes(message.loadedBytes)} / ${formatMegabytes(message.totalBytes)}）`,
      );
      return;
    }
    case 'loaded': {
      isModelLoaded = true;
      workerFailure = undefined;
      setState('idle');
      setStatus(
        `読み込み完了（${(message.elapsedMs / 1000).toFixed(1)} 秒）。録音するか音声ファイルを選んでください。`,
      );
      return;
    }
    case 'load-error': {
      // ONNX セッション作成で失敗した Worker は WebGPU デバイスや取り残されたダウンロードを抱えたままで、
      // 同じ Worker での再試行は成功しないため捨て、次の読み込みで作り直す
      discardWorker();
      setState('unloaded');
      setStatus(`モデルの読み込みに失敗しました: ${message.message}`);
      return;
    }
    case 'token': {
      pending.get(message.id)?.target.append(message.text);
      return;
    }
    case 'done': {
      pending.get(message.id)?.resolve(message.interrupted);
      pending.delete(message.id);
      return;
    }
    case 'transcribe-error': {
      pending.get(message.id)?.reject(new Error(message.message));
      pending.delete(message.id);
      return;
    }
  }
}

/**
 * Worker 自体が落ちた・読み込めなかったときは応答が二度と来ないので、待ちを全部失敗させて Worker を捨てる。
 * その場で作り直さないのは、worker.js 自体が読めない場合に作り直しと失敗が無限に繰り返されるため。
 * 次に「モデルを読み込む」を押したときに作り直す。
 */
function handleWorkerFailure(reason: string): void {
  discardWorker();
  workerFailure = reason;
  for (const { reject } of pending.values()) reject(new Error(reason));
  pending.clear();
  // 録音中・文字起こし中はその処理の終わり（settle）で読み込み前に戻るので、ここで状態を変えるのは待機系の状態だけ
  const isResting = state === 'idle' || state === 'loading';
  if (!isResting) return;
  setState('unloaded');
  setStatus(`モデルが使えなくなりました。もう一度読み込んでください（${reason}）`);
}

function spawnWorker(): Worker {
  // tsdown は worker を別エントリとして同じディレクトリに出力するので、import.meta.url からの相対で解決できる
  const spawned = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  spawned.addEventListener('message', handleWorkerMessage);
  spawned.addEventListener('error', (event) => {
    event.preventDefault();
    handleWorkerFailure(event.message || 'Worker でエラーが発生しました');
  });
  spawned.addEventListener('messageerror', () => handleWorkerFailure('Worker からのメッセージを復元できませんでした'));
  return spawned;
}

function discardWorker(): void {
  worker?.terminate();
  worker = undefined;
  isModelLoaded = false;
}

let worker: Worker | undefined;
const send = (request: WorkerRequest, transfer: Transferable[] = []): void => {
  // Worker を作るのは読み込みのときだけ。モデルの無い Worker に interrupt や transcribe を送っても意味が無いため
  const isLoadRequest = request.type === 'load';
  if (isLoadRequest) worker ??= spawnWorker();
  worker?.postMessage(request, transfer);
};

/** Worker が落ちてモデルを失っていたら、送る前に止める（新しい Worker にはモデルが無く、原因と違うエラーになるため）。 */
function assertModelAvailable(): void {
  if (isModelLoaded) return;
  throw new Error(`モデルが使えなくなりました。もう一度読み込んでください（${workerFailure ?? '不明な理由'}）`);
}

function transcribeSegment(segment: AudioSegment, target: HTMLElement): Promise<boolean> {
  const id = nextId++;
  const { promise, resolve, reject } = Promise.withResolvers<boolean>();
  pending.set(id, { target, resolve, reject });
  // 区間は元波形の subarray なので、転送で元バッファを手放さないようコピーしてから渡す
  const samples = segment.samples.slice();
  send({ type: 'transcribe', id, samples, language: languageSelect.value }, [samples.buffer]);
  return promise;
}

function appendSegmentView(segment: AudioSegment): HTMLElement {
  const row = document.createElement('div');
  row.className = 'segment';
  const time = document.createElement('span');
  time.className = 'segment-time';
  time.textContent = `${formatTimestamp(segment.startSec)}–${formatTimestamp(segment.endSec)}`;
  const text = document.createElement('span');
  text.className = 'segment-text';
  row.append(time, text);
  output.append(row);
  return text;
}

async function transcribeBlob(blob: Blob): Promise<void> {
  const canStart = state === 'idle' || state === 'finishing';
  if (!canStart) return;

  if (playerUrl !== undefined) URL.revokeObjectURL(playerUrl);
  playerUrl = URL.createObjectURL(blob);
  player.src = playerUrl;
  player.hidden = false;
  output.replaceChildren();
  isStopRequested = false;
  setState('transcribing');

  try {
    setStatus('音声をデコード中…');
    const audio = await decodeToMono16k(blob);
    const segments = splitIntoSegments(audio);
    if (segments.length === 0) {
      setStatus('音声が短すぎるか空です。');
      return;
    }

    // segments が空でないことは確認済みなので 0 にはならない
    const durationSec = audio.length / SAMPLING_RATE;
    const startedAt = performance.now();
    for (const [index, segment] of segments.entries()) {
      // 停止より先に確かめる。Worker が落ちたあとの停止を、通常の停止として報告しないため
      assertModelAvailable();
      if (isStopRequested) break;
      setStatus(`文字起こし中… ${index + 1} / ${segments.length} 区間（音声 ${durationSec.toFixed(1)} 秒）`);
      const wasInterrupted = await transcribeSegment(segment, appendSegmentView(segment));
      if (wasInterrupted) break;
    }

    const elapsedSec = (performance.now() - startedAt) / 1000;
    const outcome = isStopRequested ? '停止しました' : '完了';
    setStatus(
      `${outcome}: 音声 ${durationSec.toFixed(1)} 秒を ${elapsedSec.toFixed(1)} 秒で処理（RTF ${(elapsedSec / durationSec).toFixed(2)}）`,
    );
  } catch (error) {
    console.error(error);
    setStatus(`文字起こしに失敗しました: ${errorText(error)}`);
  } finally {
    settle();
  }
}

async function startRecording(): Promise<void> {
  setState('acquiring');
  setStatus('マイクの使用許可を待っています…');

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    settle();
    setStatus(`マイクを使えません: ${errorText(error)}`);
    return;
  }
  const releaseMicrophone = (): void => {
    for (const track of stream.getTracks()) track.stop();
  };

  const chunks: Blob[] = [];
  let activeRecorder: MediaRecorder;
  try {
    activeRecorder = new MediaRecorder(stream);
    activeRecorder.addEventListener('dataavailable', (event) => chunks.push(event.data));
    activeRecorder.addEventListener('stop', () => {
      releaseMicrophone();
      recorder = undefined;
      // マイクが抜けた・権限が取り消されたなどで録音が勝手に止まった場合も、ボタン経由と同じく文字起こしへ進める
      if (state === 'recording') setState('finishing');
      const blob = new Blob(chunks, { type: activeRecorder.mimeType });
      const isEmpty = blob.size === 0;
      if (isEmpty) {
        settle();
        setStatus('録音が空でした。');
        return;
      }
      void transcribeBlob(blob);
    });
    activeRecorder.start();
  } catch (error) {
    releaseMicrophone();
    settle();
    setStatus(`録音を開始できません: ${errorText(error)}`);
    return;
  }
  recorder = activeRecorder;
  setState('recording');
  setStatus('録音中…（もう一度押すと停止して文字起こしします）');
}

loadButton.addEventListener('click', async () => {
  const canLoad = state === 'unloaded';
  if (!canLoad) return;
  setState('loading');
  const adapter = 'gpu' in navigator ? await navigator.gpu.requestAdapter() : null;
  const isStillLoading = state === 'loading';
  if (!isStillLoading) return;
  const hasWebGpu = adapter !== null;
  if (!hasWebGpu) {
    setState('unloaded');
    setStatus('WebGPU が使えません。デスクトップ版の Chrome / Edge の最新版で開いてください。');
    return;
  }
  progressBar.value = 0;
  setStatus('モデルを準備中…（2 回目以降はブラウザのキャッシュから読み込みます）');
  send({ type: 'load' });
});

recordButton.addEventListener('click', () => {
  if (state === 'recording') {
    setState('finishing');
    setStatus('録音を終了しています…');
    recorder?.stop();
    return;
  }
  if (state !== 'idle') return;
  void startRecording();
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file === undefined) return;
  fileInput.value = '';
  void transcribeBlob(file);
});

stopButton.addEventListener('click', () => {
  const isTranscribing = state === 'transcribing';
  if (!isTranscribing) return;
  isStopRequested = true;
  stopButton.disabled = true;
  send({ type: 'interrupt' });
});

setState('unloaded');
