import type { WorkerRequest, WorkerResponse } from '../protocol.ts';

type Send = (request: WorkerRequest, transfer: Transferable[]) => void;

/**
 * Worker との protocol.ts のやり取りを Promise にまとめる。main.ts のメッセージ処理（id ごとのトークン連結・done・transcribe-error）と同じ規則。
 * Worker そのものではなく送信関数を受け取り、受信は handleMessage / fail で外から流し込む。
 * Worker を直接抱えないのは、ブラウザ無しの vitest で応答の順序や失敗を再現して検証できるようにするため
 */
export class TranscriptionClient {
  readonly #send: Send;
  readonly #now: () => number;
  /** 区間 ID ごとの、連結中のテキストと完了待ち。 */
  readonly #pending = new Map<
    number,
    { text: string; resolve: (text: string) => void; reject: (error: Error) => void }
  >();
  #nextId = 0;
  #load: { startedAt: number; resolve: (ms: number) => void; reject: (error: Error) => void } | undefined;
  #failure: string | null = null;

  /** ダウンロードの進捗（%）。UI の表示用。 */
  onLoadProgress: ((progress: number) => void) | undefined;

  constructor(send: Send, now: () => number = () => performance.now()) {
    this.#send = send;
    this.#now = now;
  }

  /** Worker が落ちた理由。null でなければ以後の要求はすべて失敗する。 */
  get failure(): string | null {
    return this.#failure;
  }

  /**
   * モデルを読み込み、送信から 'loaded' までの経過ミリ秒を返す。
   * Worker の elapsedMs ではなくこちらで測るのは、worker.js の取得・評価と ORT 初期化も含めた「利用者が待つ時間」を比べるため
   */
  load(modelKey: string): Promise<number> {
    if (this.#failure !== null) return Promise.reject(new Error(this.#failure));
    if (this.#load !== undefined) return Promise.reject(new Error('モデルを読み込み中です'));
    const { promise, resolve, reject } = Promise.withResolvers<number>();
    this.#load = { startedAt: this.#now(), resolve, reject };
    this.#send({ type: 'load', modelKey }, []);
    return promise;
  }

  /** 1 区間を文字起こしし、区間内のトークンを連結したテキストを返す。 */
  transcribe(samples: Float32Array, language: string): Promise<string> {
    if (this.#failure !== null) return Promise.reject(new Error(this.#failure));
    const id = this.#nextId++;
    const { promise, resolve, reject } = Promise.withResolvers<string>();
    this.#pending.set(id, { text: '', resolve, reject });
    // 区間は元波形の subarray なので、転送で元バッファを手放さないようコピーしてから渡す
    const copy = samples.slice();
    this.#send({ type: 'transcribe', id, samples: copy, language }, [copy.buffer]);
    return promise;
  }

  handleMessage(message: WorkerResponse): void {
    switch (message.type) {
      case 'load-progress': {
        this.onLoadProgress?.(message.progress);
        return;
      }
      case 'loaded': {
        const load = this.#load;
        this.#load = undefined;
        load?.resolve(this.#now() - load.startedAt);
        return;
      }
      case 'load-error': {
        const load = this.#load;
        this.#load = undefined;
        load?.reject(new Error(message.message));
        return;
      }
      case 'token': {
        const entry = this.#pending.get(message.id);
        if (entry !== undefined) entry.text += message.text;
        return;
      }
      case 'done': {
        // 評価では interrupt を送らないので interrupted は見ない
        const entry = this.#pending.get(message.id);
        this.#pending.delete(message.id);
        entry?.resolve(entry.text);
        return;
      }
      case 'transcribe-error': {
        this.#pending.get(message.id)?.reject(new Error(message.message));
        this.#pending.delete(message.id);
        return;
      }
    }
  }

  /**
   * Worker 自体が落ちたときに呼ぶ。応答は二度と来ないので、待ちを全部失敗させ、以後の要求も即座に失敗させる。
   * 作り直して続行しないのは、同じ原因（GPU メモリ不足など）で落ち続け、失敗が数千件並ぶだけになるため
   */
  fail(reason: string): void {
    this.#failure ??= reason;
    this.#load?.reject(new Error(reason));
    this.#load = undefined;
    for (const { reject } of this.#pending.values()) reject(new Error(reason));
    this.#pending.clear();
  }
}
