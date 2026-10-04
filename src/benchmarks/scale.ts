export interface LinearScale {
  /** 軸の右端（0 起点）。 */
  max: number;
  /** 目盛りの値（0 と max を含む）。 */
  ticks: number[];
}

/**
 * 0 起点の軸の目盛りを、1・2・5 × 10^k の切りのよい刻みで作る。
 * 最大値がちょうど目盛りに乗るときもそのまま使い、棒が軸の端に貼り付いても値ラベルは棒の外に出す。
 * Why not d3-scale の nice(): 0 起点・1 軸だけの単純な用途に依存を足すほどではないため
 */
export function niceScale(maxValue: number, targetTicks = 4): LinearScale {
  const isEmpty = !Number.isFinite(maxValue) || maxValue <= 0;
  if (isEmpty) return { max: 1, ticks: [0, 1] };

  const rawStep = maxValue / targetTicks;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const multiplier = [1, 2, 5, 10].find((candidate) => normalized <= candidate) ?? 10;
  const step = multiplier * magnitude;
  const max = Math.ceil(maxValue / step - 1e-9) * step;

  const ticks: number[] = [];
  const count = Math.round(max / step);
  // 浮動小数の誤差（0.30000000000000004 など）を目盛りに出さないよう、刻みの桁で丸める
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  for (let index = 0; index <= count; index++) ticks.push(Number((index * step).toFixed(decimals)));
  return { max: Number(max.toFixed(decimals)), ticks };
}
