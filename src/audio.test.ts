import { describe, expect, it } from 'vitest';

import {
  downmixToMono,
  formatTimestamp,
  MAX_SEGMENT_SECONDS,
  MIN_SEGMENT_SECONDS,
  SAMPLING_RATE,
  splitIntoSegments,
} from './audio.ts';

describe('downmixToMono', () => {
  it('モノラル入力はコピーせずそのまま返す', () => {
    const mono = new Float32Array([0.1, 0.2]);
    expect(downmixToMono([mono])).toBe(mono);
  });

  it('ステレオは左右の平均になる', () => {
    const result = downmixToMono([new Float32Array([1, 0.5]), new Float32Array([0, -0.5])]);
    expect(Array.from(result)).toEqual([0.5, 0]);
  });

  it('チャンネルが無ければ空配列を返す', () => {
    expect(downmixToMono([])).toHaveLength(0);
  });
});

describe('splitIntoSegments', () => {
  it('Gemma 4 の上限（30 秒）を超える区間を作らない', () => {
    const audio = new Float32Array(SAMPLING_RATE * 75);
    const segments = splitIntoSegments(audio);
    expect(segments.map((s) => [s.startSec, s.endSec])).toEqual([
      [0, 30],
      [30, 60],
      [60, 75],
    ]);
    for (const segment of segments) {
      expect(segment.samples.length).toBeLessThanOrEqual(SAMPLING_RATE * MAX_SEGMENT_SECONDS);
    }
  });

  it('全区間をつなぐと元の波形を欠けなく覆う', () => {
    const audio = Float32Array.from({ length: 10 }, (_, i) => i);
    const segments = splitIntoSegments(audio, 2, 2, 0);
    const joined = segments.flatMap((s) => Array.from(s.samples));
    expect(joined).toEqual(Array.from(audio));
  });

  it('上限ちょうどの長さは 1 区間になる', () => {
    expect(splitIntoSegments(new Float32Array(SAMPLING_RATE * MAX_SEGMENT_SECONDS))).toHaveLength(1);
  });

  it('音声トークンを作れないほど短い末尾の区間は捨てる', () => {
    const tail = Math.ceil(SAMPLING_RATE * MIN_SEGMENT_SECONDS) - 1;
    const segments = splitIntoSegments(new Float32Array(SAMPLING_RATE * MAX_SEGMENT_SECONDS + tail));
    expect(segments).toHaveLength(1);
  });

  it('最短長ちょうどの末尾の区間は残す', () => {
    const tail = Math.ceil(SAMPLING_RATE * MIN_SEGMENT_SECONDS);
    const segments = splitIntoSegments(new Float32Array(SAMPLING_RATE * MAX_SEGMENT_SECONDS + tail));
    expect(segments).toHaveLength(2);
  });

  it('空の波形からは区間を作らない', () => {
    expect(splitIntoSegments(new Float32Array(0))).toEqual([]);
  });
});

describe('formatTimestamp', () => {
  it.each([
    [0, '00:00'],
    [8.98, '00:09'],
    [59.4, '00:59'],
    [61, '01:01'],
    [3600, '60:00'],
    [-1, '00:00'],
  ])('%d 秒を %s と表示する', (seconds, expected) => {
    expect(formatTimestamp(seconds)).toBe(expected);
  });
});
