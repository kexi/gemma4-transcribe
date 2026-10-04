import { describe, expect, it } from 'vitest';

import type { WorkerRequest } from '../protocol.ts';
import { TranscriptionClient } from './client.ts';

function setup(): { client: TranscriptionClient; sent: WorkerRequest[]; clock: { now: number } } {
  const sent: WorkerRequest[] = [];
  const clock = { now: 0 };
  const client = new TranscriptionClient(
    (request) => sent.push(request),
    () => clock.now,
  );
  return { client, sent, clock };
}

/** 直前に送った transcribe 要求。 */
function lastTranscribe(sent: readonly WorkerRequest[]): Extract<WorkerRequest, { type: 'transcribe' }> {
  const request = sent.at(-1);
  if (request?.type !== 'transcribe') throw new Error('transcribe が送られていません');
  return request;
}

describe('TranscriptionClient.load', () => {
  it('load 要求を送り、送信から loaded までの経過時間で解決する', async () => {
    const { client, sent, clock } = setup();
    clock.now = 100;
    const loading = client.load('e2b');
    expect(sent).toEqual([{ type: 'load', modelKey: 'e2b' }]);
    clock.now = 1600;
    client.handleMessage({ type: 'loaded', elapsedMs: 1 });
    await expect(loading).resolves.toBe(1500);
  });

  it('load-error で失敗する', async () => {
    const { client } = setup();
    const loading = client.load('e2b');
    client.handleMessage({ type: 'load-error', message: 'OOM' });
    await expect(loading).rejects.toThrow('OOM');
  });

  it('ダウンロードの進捗を通知する', () => {
    const { client } = setup();
    const progress: number[] = [];
    client.onLoadProgress = (value) => progress.push(value);
    client.handleMessage({ type: 'load-progress', progress: 42, loadedBytes: 1, totalBytes: 2 });
    expect(progress).toEqual([42]);
  });
});

describe('TranscriptionClient.transcribe', () => {
  it('同じ id のトークンを順に連結し、done で解決する（他の id のトークンは混ざらない）', async () => {
    const { client, sent } = setup();
    const first = client.transcribe(new Float32Array(4), 'Japanese');
    const firstId = lastTranscribe(sent).id;
    const second = client.transcribe(new Float32Array(4), 'Japanese');
    const secondId = lastTranscribe(sent).id;

    client.handleMessage({ type: 'token', id: firstId, text: 'こん' });
    client.handleMessage({ type: 'token', id: secondId, text: '別' });
    client.handleMessage({ type: 'token', id: firstId, text: 'にちは' });
    client.handleMessage({ type: 'done', id: firstId, interrupted: false });
    client.handleMessage({ type: 'done', id: secondId, interrupted: false });

    await expect(first).resolves.toBe('こんにちは');
    await expect(second).resolves.toBe('別');
  });

  it('区間の波形をコピーして送り、呼び出し側の配列を転送で失わない', () => {
    const { client, sent } = setup();
    const audio = new Float32Array([0.1, 0.2, 0.3]);
    void client.transcribe(audio.subarray(1), 'Japanese');
    const request = lastTranscribe(sent);
    expect(Array.from(request.samples)).toEqual([audio[1], audio[2]]);
    expect(request.samples.buffer).not.toBe(audio.buffer);
    expect(request.language).toBe('Japanese');
  });

  it('transcribe-error で、その区間だけ失敗する', async () => {
    const { client, sent } = setup();
    const failing = client.transcribe(new Float32Array(1), 'Japanese');
    const failingId = lastTranscribe(sent).id;
    const succeeding = client.transcribe(new Float32Array(1), 'Japanese');
    const succeedingId = lastTranscribe(sent).id;

    client.handleMessage({ type: 'transcribe-error', id: failingId, message: 'boom' });
    client.handleMessage({ type: 'done', id: succeedingId, interrupted: false });

    await expect(failing).rejects.toThrow('boom');
    await expect(succeeding).resolves.toBe('');
    expect(client.failure).toBeNull();
  });
});

describe('TranscriptionClient.fail', () => {
  it('待ち中の要求をすべて失敗させ、以後の要求も送らずに失敗させる', async () => {
    const { client, sent } = setup();
    const loading = client.load('e2b');
    const transcribing = client.transcribe(new Float32Array(1), 'Japanese');

    client.fail('Worker crashed');

    await expect(loading).rejects.toThrow('Worker crashed');
    await expect(transcribing).rejects.toThrow('Worker crashed');
    expect(client.failure).toBe('Worker crashed');

    const sentBefore = sent.length;
    await expect(client.transcribe(new Float32Array(1), 'Japanese')).rejects.toThrow('Worker crashed');
    await expect(client.load('e2b')).rejects.toThrow('Worker crashed');
    expect(sent).toHaveLength(sentBefore);
  });

  it('最初の失敗理由を保持する', () => {
    const { client } = setup();
    client.fail('first');
    client.fail('second');
    expect(client.failure).toBe('first');
  });
});
