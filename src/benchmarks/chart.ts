import { formatCount, formatPercent, PERCENT_DIGITS, seriesClass, shortModelLabels } from './format.ts';
import type { LinearScale } from './scale.ts';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** 1 本の棒（モデル）。value は % 単位（12.3 = 12.3%）。 */
export interface BarDatum {
  modelKey: string;
  modelLabel: string;
  value: number | null;
  n: number | null;
  failed: number | null;
  median: number | null;
  outliers: number | null;
}

export interface BarChartSpec {
  title: string;
  subtitle: string;
  bars: readonly BarDatum[];
  /** small multiples 全体で共有する軸（グラフ間で棒の長さを比べられるように）。 */
  scale: LinearScale;
  /**
   * ラベル列と値ラベル欄の幅を測る対象（small multiples の全グラフの棒）。
   * Why not 各グラフの棒だけで測る: 失敗の注記が付くグラフだけ描画幅が縮み、同じ軸の目盛りの位置がグラフごとにずれるため
   */
  layoutBars: readonly BarDatum[];
}

/** 行の高さと棒の太さ。棒は細め（24px 以下）にして行の残りは余白にする。 */
const ROW_HEIGHT = 30;
const BAR_THICKNESS = 14;
/** 値のある端だけを丸める半径。 */
const BAR_RADIUS = 4;
const AXIS_HEIGHT = 22;
const LABEL_GAP = 8;
const VALUE_GAP = 6;
const FONT_SIZE = 12;
/** 狭い画面でも棒が潰れないための最小の描画幅。 */
const MIN_PLOT_WIDTH = 60;

const svg = <K extends keyof SVGElementTagNameMap>(
  name: K,
  attributes: Record<string, string | number>,
): SVGElementTagNameMap[K] => {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
};

let measureContext: CanvasRenderingContext2D | null | undefined;

/** SVG の text と同じフォントで文字幅を測る（ラベル列と値ラベルの幅を、はみ出さないよう先に決めるため）。 */
function textWidth(text: string, font: string): number {
  measureContext ??= document.createElement('canvas').getContext('2d');
  if (!measureContext) return text.length * FONT_SIZE;
  measureContext.font = font;
  return measureContext.measureText(text).width;
}

/** 0 起点の横棒。値のある右端だけ角を丸め、基線側は四角のまま。 */
function barPath(x: number, y: number, width: number, height: number): string {
  const radius = Math.min(BAR_RADIUS, width, height / 2);
  const right = x + width;
  return [
    `M${x},${y}`,
    `H${right - radius}`,
    `A${radius},${radius} 0 0 1 ${right},${y + radius}`,
    `V${y + height - radius}`,
    `A${radius},${radius} 0 0 1 ${right - radius},${y + height}`,
    `H${x}`,
    'Z',
  ].join(' ');
}

/** 棒の横に出す文字列。失敗があれば必ず併記し、値が無ければ理由を出す。 */
export function barValueLabel(bar: BarDatum): { value: string; note: string } {
  const failed = bar.failed ?? 0;
  const failedText = failed > 0 ? `失敗 ${formatCount(failed)} 件` : '';
  if (bar.value !== null) return { value: `${bar.value.toFixed(PERCENT_DIGITS)}%`, note: failedText };
  const isAllFailed = failed > 0;
  return { value: isAllFailed ? '計測不可' : '未計測', note: failedText };
}

export interface Tooltip {
  show(bar: BarDatum, title: string, anchor: { x: number; y: number }): void;
  hide(): void;
}

/** ツールチップ 1 つを全グラフで共有する。文字はすべて textContent で入れる（JSON の文字列を HTML として解釈しない）。 */
export function createTooltip(): Tooltip {
  const root = document.createElement('div');
  root.className = 'bench-tooltip';
  root.setAttribute('role', 'tooltip');
  root.hidden = true;
  document.body.append(root);

  return {
    show(bar, title, anchor) {
      const { value, note } = barValueLabel(bar);
      const valueLine = document.createElement('strong');
      valueLine.className = 'bench-tooltip-value';
      valueLine.textContent = value;
      const key = document.createElement('span');
      key.className = `bench-tooltip-key ${seriesClass(bar.modelKey)}`;
      const name = document.createElement('div');
      name.className = 'bench-tooltip-name';
      name.append(key, document.createTextNode(bar.modelLabel));
      const details = document.createElement('div');
      details.className = 'bench-tooltip-details';
      const parts = [
        title,
        `${formatCount(bar.n)} クリップ`,
        note,
        bar.median === null ? '' : `中央値 ${formatPercent(bar.median)}`,
        bar.outliers === null || bar.outliers === 0 ? '' : `外れ値 ${formatCount(bar.outliers)} 件`,
      ].filter((part) => part !== '');
      details.textContent = parts.join(' · ');
      root.replaceChildren(valueLine, name, details);
      root.hidden = false;

      // 画面端ではみ出さないよう、ポインタの右下を基本に収まらなければ反対側へ寄せる
      const margin = 12;
      const { width, height } = root.getBoundingClientRect();
      const fitsRight = anchor.x + margin + width <= window.innerWidth - 4;
      const left = fitsRight ? anchor.x + margin : Math.max(4, anchor.x - margin - width);
      const fitsBelow = anchor.y + margin + height <= window.innerHeight - 4;
      const top = fitsBelow ? anchor.y + margin : Math.max(4, anchor.y - margin - height);
      root.style.left = `${left}px`;
      root.style.top = `${top}px`;
    },
    hide() {
      root.hidden = true;
    },
  };
}

/** 1 つのグラフ（データセット 1 つ分）を幅 width の SVG として描く。 */
export function renderBarChart(spec: BarChartSpec, width: number, font: string, tooltip: Tooltip): SVGSVGElement {
  // 短いラベルの重なりは全グラフの棒（layoutBars）で判定し、どのグラフでも同じモデルは同じ名前にする
  const fullLabels = spec.layoutBars.map((bar) => bar.modelLabel);
  const layoutLabels = shortModelLabels(fullLabels);
  const shortOf = new Map(fullLabels.map((label, index) => [label, layoutLabels[index] ?? label]));
  const labels = spec.bars.map((bar) => shortOf.get(bar.modelLabel) ?? bar.modelLabel);
  const valueTexts = spec.layoutBars.map((bar) => {
    const { value, note } = barValueLabel(bar);
    return note === '' ? value : `${value} · ${note}`;
  });
  // ラベル列は最長のラベルに合わせ、ただし幅の 40% までに抑える（残りを棒に使う）
  const labelWidth = Math.min(
    Math.max(...layoutLabels.map((label) => textWidth(label, font)), 0) + LABEL_GAP,
    width * 0.4,
  );
  const valueWidth = Math.max(...valueTexts.map((text) => textWidth(text, font)), 0) + VALUE_GAP + 2;
  const plotLeft = Math.ceil(labelWidth);
  const plotWidth = Math.max(width - plotLeft - valueWidth, MIN_PLOT_WIDTH);
  const plotHeight = spec.bars.length * ROW_HEIGHT;
  const height = plotHeight + AXIS_HEIGHT;
  const xOf = (value: number): number => plotLeft + (value / spec.scale.max) * plotWidth;

  const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, class: 'bench-chart-svg' });
  root.setAttribute('role', 'group');
  root.setAttribute('aria-label', `${spec.title}の CER（正規化）`);

  const grid = svg('g', { class: 'bench-grid', 'aria-hidden': 'true' });
  for (const tick of spec.scale.ticks) {
    const x = Math.round(xOf(tick)) + 0.5;
    grid.append(svg('line', { x1: x, x2: x, y1: 0, y2: plotHeight, class: tick === 0 ? 'bench-baseline' : '' }));
    const label = svg('text', { x, y: plotHeight + 15, 'text-anchor': 'middle', class: 'bench-axis-label' });
    label.textContent = `${tick}%`;
    grid.append(label);
  }
  root.append(grid);

  spec.bars.forEach((bar, index) => {
    const top = index * ROW_HEIGHT;
    const centerY = top + ROW_HEIGHT / 2;
    const row = svg('g', { class: 'bench-bar-row', tabindex: 0 });
    const { value, note } = barValueLabel(bar);
    row.setAttribute('aria-label', [bar.modelLabel, value, note].filter((part) => part !== '').join('、'));

    // ヒット領域は行全体（棒より大きく）。透明でも pointer イベントを受けるよう fill を指定する
    row.append(svg('rect', { x: 0, y: top, width, height: ROW_HEIGHT, class: 'bench-hit' }));

    const nameLabel = svg('text', {
      x: plotLeft - LABEL_GAP,
      y: centerY,
      'text-anchor': 'end',
      'dominant-baseline': 'central',
      class: 'bench-row-label',
    });
    nameLabel.textContent = labels[index] ?? bar.modelLabel;
    row.append(nameLabel);

    const hasValue = bar.value !== null;
    const barWidth = hasValue ? Math.max(xOf(bar.value ?? 0) - plotLeft, 0) : 0;
    const hasVisibleBar = barWidth > 0;
    if (hasVisibleBar) {
      const barY = centerY - BAR_THICKNESS / 2;
      row.append(
        svg('path', {
          d: barPath(plotLeft, barY, barWidth, BAR_THICKNESS),
          class: `bench-bar ${seriesClass(bar.modelKey)}`,
        }),
      );
    }

    const valueLabel = svg('text', {
      x: plotLeft + barWidth + VALUE_GAP,
      y: centerY,
      'dominant-baseline': 'central',
      class: hasValue ? 'bench-value-label' : 'bench-value-label is-missing',
    });
    valueLabel.textContent = value;
    if (note !== '') {
      const noteSpan = svg('tspan', { class: 'bench-value-note' });
      noteSpan.textContent = ` · ${note}`;
      valueLabel.append(noteSpan);
    }
    row.append(valueLabel);

    row.addEventListener('pointermove', (event) =>
      tooltip.show(bar, spec.title, { x: event.clientX, y: event.clientY }),
    );
    row.addEventListener('pointerleave', () => tooltip.hide());
    row.addEventListener('focus', () => {
      const rect = row.getBoundingClientRect();
      tooltip.show(bar, spec.title, { x: rect.left + plotLeft + barWidth, y: rect.top + ROW_HEIGHT / 2 });
    });
    row.addEventListener('blur', () => tooltip.hide());
    root.append(row);
  });

  return root;
}
