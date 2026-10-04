import { describe, expect, it } from 'vitest';

import { buildHeadline, joinLabels, withParticle } from './headline.ts';
import { fixtureResults, metrics, model, withModels } from './testing.ts';

describe('buildHeadline', () => {
  it('精度の最良と速度の最速が別のモデルなら両方を言い、最速の値が失敗を除いたものなら添える（フィクスチャ）', () => {
    expect(buildHeadline(fixtureResults())).toBe(
      '日本語の総合 CER（正規化）は Gemma 4 E4B（q4f16）が最良（15.9%）、速度は Gemma 4 E2B QAT mobile（q2f16）が最速（RTF 0.111、失敗 2 件を除く）。',
    );
  });

  it('最良と最速が同じモデルならまとめて言う', () => {
    const results = withModels([model('a', 'A', metrics(0.1, 0.1)), model('b', 'B', metrics(0.2, 0.2))]);
    expect(buildHeadline(results)).toBe('日本語の総合 CER（正規化）は A が最良（10.0%）で、速度も最速（RTF 0.100）。');
  });

  it('表示桁で同じ CER は同率として並べ、勝者を 1 つに決め打ちしない', () => {
    const results = withModels([model('a', 'A', metrics(0.10004, 0.2)), model('b', 'B', metrics(0.1, 0.1))]);
    expect(buildHeadline(results)).toBe(
      '日本語の総合 CER（正規化）は A と B が同率で最良（10.0%）、速度は B が最速（RTF 0.100）。',
    );
  });

  it('画面で同じ「2.1%」に見える 2.15% と 2.1% は同率として並べる（表示と同じ toFixed の丸めで比べる）', () => {
    const results = withModels([model('a', 'A', metrics(0.0215, 0.2)), model('b', 'B', metrics(0.021, 0.1))]);
    expect(buildHeadline(results)).toContain('A と B が同率で最良（2.1%）');
  });

  it('全角の括弧で終わる名前の同率は「と」の前に空白を挟まない', () => {
    const results = withModels([
      model('a', 'Gemma 4 E2B（q4f16）', metrics(0.1, 0.2)),
      model('b', 'Gemma 4 E4B（q4f16）', metrics(0.1, 0.1)),
    ]);
    expect(buildHeadline(results)).toContain('Gemma 4 E2B（q4f16）と Gemma 4 E4B（q4f16）が同率で最良（10.0%）');
  });

  it('最良モデルに失敗クリップがあれば、失敗を除いた値であることを添える', () => {
    const results = withModels([model('a', 'A', metrics(0.1, 0.2, { failed: 3 })), model('b', 'B', metrics(0.2, 0.1))]);
    expect(buildHeadline(results)).toContain('A が最良（10.0%、失敗 3 件を除く）');
  });

  it('未計測のモデルは比較に入れない', () => {
    const results = withModels([
      model('a', 'A', null),
      model('b', 'B', metrics(0.2, 0.1)),
      model('c', 'C', metrics(0.3, 0.2)),
    ]);
    expect(buildHeadline(results)).toContain('B が最良（20.0%）');
  });

  it('未実行のクリップがあるモデルは CER が良くても最良・最速にせず、比較から外したことを言う', () => {
    const results = withModels([
      model('a', 'A', metrics(0.05, 0.05, { n: 120 }), { missing: 180 }),
      model('b', 'B', metrics(0.2, 0.1)),
      model('c', 'C', metrics(0.3, 0.2)),
    ]);
    expect(buildHeadline(results)).toBe(
      '日本語の総合 CER（正規化）は B が最良（20.0%）で、速度も最速（RTF 0.100）。A は未完了のため比較から除外しています。',
    );
  });

  it('missing が 0 でも件数が他と違う（全クリップ数に届かない）モデルは比較から外す', () => {
    const results = withModels([
      model('a', 'A', metrics(0.05, 0.05, { n: 100 })),
      model('b', 'B', metrics(0.2, 0.1)),
      model('c', 'C', metrics(0.3, 0.2)),
    ]);
    expect(buildHeadline(results)).toContain('B が最良（20.0%）');
    expect(buildHeadline(results)).toContain('A は未完了のため比較から除外しています。');
  });

  it('走り終えたモデルが 1 つだけなら「最良」と言わず、他が未完了で比べていないことを言う', () => {
    const results = withModels([
      model('a', 'A', metrics(0.2, 0.1)),
      model('b', 'B', metrics(0.05, 0.05, { n: 120 }), { missing: 180 }),
    ]);
    expect(buildHeadline(results)).toBe(
      '日本語の総合 CER（正規化）は A で 20.0%（300 クリップ）。B は未完了のため比較から除外しています。',
    );
  });

  it('走り終えたモデルが 1 つも無ければ、最良を言わずに未完了のモデルを挙げる', () => {
    const results = withModels([model('a', 'A', metrics(0.2, 0.1, { n: 10 }), { missing: 290 })]);
    expect(buildHeadline(results)).toBe(
      '全クリップの計測を終えたモデルはまだありません。A は未完了のため比較から除外しています。',
    );
  });

  it('計測済みが 1 モデルだけなら「最良」とは言わず、比較対象が無いことを言う', () => {
    const results = withModels([model('a', 'A', metrics(0.204, 0.145)), model('b', 'B', null)]);
    expect(buildHeadline(results)).toBe(
      '日本語の総合 CER（正規化）は A で 20.4%（300 クリップ）。比較対象のモデルはまだ計測されていません。',
    );
  });

  it('計測済みのモデルが無ければその旨を言う', () => {
    expect(buildHeadline(withModels([model('a', 'A', null)]))).toBe('まだ計測済みのモデルがありません。');
    expect(buildHeadline(withModels([]))).toBe('まだ計測済みのモデルがありません。');
  });

  it('RTF が 1 モデルにしか無ければ速度には触れない', () => {
    const results = withModels([model('a', 'A', metrics(0.1, null)), model('b', 'B', metrics(0.2, 0.1))]);
    expect(buildHeadline(results)).toBe('日本語の総合 CER（正規化）は A が最良（10.0%）。');
  });
});

describe('withParticle', () => {
  it('英数字で終わる名前には半角空白を挟んで助詞を続ける', () => {
    expect(withParticle('Gemma 4 E2B', 'が')).toBe('Gemma 4 E2B が');
  });

  it('全角の括弧で終わる名前には空白を挟まない', () => {
    expect(withParticle('Gemma 4 E2B（q4f16）', 'が')).toBe('Gemma 4 E2B（q4f16）が');
  });
});

describe('joinLabels', () => {
  it('英数字で終わる名前は「 と 」でつなぐ', () => {
    expect(joinLabels(['A', 'B', 'C'])).toBe('A と B と C');
  });

  it('全角の括弧で終わる名前の後の「と」には空白を挟まず、英数字で始まる次の名前の前にだけ挟む', () => {
    expect(joinLabels(['E2B（q4f16）', 'E4B（q4f16）'])).toBe('E2B（q4f16）と E4B（q4f16）');
    expect(joinLabels(['E2B（q4f16）', '軽量版'])).toBe('E2B（q4f16）と軽量版');
  });
});
