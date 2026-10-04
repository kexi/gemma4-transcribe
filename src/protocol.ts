/** メインスレッド → Worker のメッセージ。 */
export type WorkerRequest =
  | { type: 'load' }
  | { type: 'transcribe'; id: number; samples: Float32Array; language: string }
  | { type: 'interrupt' };

/** Worker → メインスレッドのメッセージ。 */
export type WorkerResponse =
  | { type: 'load-progress'; progress: number; loadedBytes: number; totalBytes: number }
  | { type: 'loaded'; elapsedMs: number }
  | { type: 'load-error'; message: string }
  | { type: 'token'; id: number; text: string }
  | { type: 'done'; id: number; interrupted: boolean }
  | { type: 'transcribe-error'; id: number; message: string };
