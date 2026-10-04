import { describe, expect, it } from 'vitest';

import { MODELS } from '../models.ts';
import {
  bestIndices,
  fixedKey,
  formatCount,
  formatFixed,
  formatPercent,
  MISSING,
  percentKey,
  seriesClass,
  seriesSlot,
  shortCommit,
  shortModelLabel,
  shortModelLabels,
} from './format.ts';

describe('formatPercent', () => {
  it('割合を小数 1 桁の % にする', () => {
    expect(formatPercent(0.2054)).toBe('20.5%');
    expect(formatPercent(0)).toBe('0.0%');
  });

  it('100% を超える CER（挿入の多いクリップ）もそのまま出す', () => {
    expect(formatPercent(3)).toBe('300.0%');
  });

  it('値が無ければ「—」にする', () => {
    expect(formatPercent(null)).toBe(MISSING);
  });
});

describe('formatFixed / formatCount', () => {
  it('指定桁の固定小数にし、無ければ「—」', () => {
    expect(formatFixed(0.14523, 3)).toBe('0.145');
    expect(formatFixed(null, 3)).toBe(MISSING);
  });

  it('件数は整数に丸めて桁区切りを付け、無ければ「—」', () => {
    expect(formatCount(5263)).toBe('5,263');
    expect(formatCount(0)).toBe('0');
    expect(formatCount(null)).toBe(MISSING);
  });
});

describe('shortCommit', () => {
  it('SHA は 7 桁に縮め、SHA でない文字列はそのまま返す', () => {
    expect(shortCommit('6ccfadf46a87975d37372efcc9316cb2eaf5cf25')).toBe('6ccfadf');
    expect(shortCommit('unknown')).toBe('unknown');
  });
});

describe('shortModelLabel', () => {
  it.each([
    ['Gemma 4 E2B（q4f16）', 'E2B'],
    ['Gemma 4 E2B QAT mobile（q2f16）', 'E2B QAT mobile'],
    ['Gemma 4 E4B (q4f16)', 'E4B'],
    ['Whisper large-v3', 'Whisper large-v3'],
  ])('%s をグラフの行ラベル用に %s へ縮める', (label, expected) => {
    expect(shortModelLabel(label)).toBe(expected);
  });

  it('縮めると空になるラベルは元のまま返す', () => {
    expect(shortModelLabel('Gemma 4 （x）')).toBe('Gemma 4 （x）');
  });
});

describe('shortModelLabels', () => {
  it('重ならなければ量子化の括弧まで落とした短い名前にする', () => {
    expect(shortModelLabels(['Gemma 4 E2B（q4f16）', 'Gemma 4 E2B QAT mobile（q2f16）'])).toEqual([
      'E2B',
      'E2B QAT mobile',
    ]);
  });

  it('量子化だけが違うモデルは括弧を残し、色だけで区別させない', () => {
    expect(shortModelLabels(['Gemma 4 E2B（q4f16）', 'Gemma 4 E2B（q4）', 'Gemma 4 E4B（q4f16）'])).toEqual([
      'E2B（q4f16）',
      'E2B（q4）',
      'E4B',
    ]);
  });

  it('括弧を残しても重なるなら元のラベルのまま出す', () => {
    expect(shortModelLabels(['Gemma 4 E2B（q4f16）', 'E2B（q4f16）'])).toEqual([
      'Gemma 4 E2B（q4f16）',
      'E2B（q4f16）',
    ]);
  });

  it('同じモデルが何度出ても（全グラフの棒）重なりとは数えず、入力と同じ順で返す', () => {
    const labels = ['Gemma 4 E2B（q4f16）', 'Gemma 4 E4B（q4f16）', 'Gemma 4 E2B（q4f16）'];
    expect(shortModelLabels(labels)).toEqual(['E2B', 'E4B', 'E2B']);
  });
});

describe('bestIndices', () => {
  const rtfKey = fixedKey(3);

  it('最小値の添字を返す（CER・RTF は小さいほど良い）', () => {
    expect(bestIndices([0.2, 0.15, 0.3], rtfKey)).toEqual(new Set([1]));
  });

  it('表示桁で丸めて同じになる値は同率として両方返す', () => {
    expect(bestIndices([0.15049, 0.1501, 0.3], rtfKey)).toEqual(new Set([0, 1]));
  });

  it('% 表示の比較は formatPercent と同じ丸めで行い、どちらも「2.1%」と出る 2.15% と 2.1% を同率にする', () => {
    expect([formatPercent(0.0215), formatPercent(0.021)]).toEqual(['2.1%', '2.1%']);
    expect(bestIndices([0.0215, 0.021], percentKey)).toEqual(new Set([0, 1]));
  });

  it('null（未計測）は比較から除く', () => {
    expect(bestIndices([null, 0.3, 0.2], rtfKey)).toEqual(new Set([2]));
  });

  it('比べる相手がいない（値が 1 つ以下）なら強調しない', () => {
    expect(bestIndices([null, 0.3, null], rtfKey)).toEqual(new Set());
    expect(bestIndices([], rtfKey)).toEqual(new Set());
  });
});

describe('seriesSlot', () => {
  it('MODELS の先頭 4 モデルに slot 1–4 を key で固定して割り当てる（e2b→1, e2b-qat→2, e4b→3, e4b-qat→4）', () => {
    expect(['e2b', 'e2b-qat', 'e4b', 'e4b-qat'].map(seriesSlot)).toEqual([1, 2, 3, 4]);
    expect(MODELS.slice(0, 4).map((model) => model.key)).toEqual(['e2b', 'e2b-qat', 'e4b', 'e4b-qat']);
  });

  it('MODELS に無いモデルには色を生成せず中立色（null）にする', () => {
    expect(seriesSlot('streaming-x')).toBeNull();
    expect(seriesClass('streaming-x')).toBe('bench-series-other');
  });

  it('CSS クラス名は slot 番号から作る', () => {
    expect(seriesClass('e4b')).toBe('bench-series-3');
  });
});
