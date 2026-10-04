/** Gemma 4 の音声エンコーダが前提とするサンプリングレート（Hz）。 */
export const SAMPLING_RATE = 16_000;

/**
 * 1 回の推論に渡せる音声の最大長（秒）。
 * Gemma 4 の音声入力は最大 750 トークン × 40ms = 30 秒で、超えた分は processor が黙って切り捨てるため、
 * 呼び出し側で区間に分けて順に推論する。
 */
export const MAX_SEGMENT_SECONDS = 30;

/**
 * これより短い末尾の区間は捨てる（秒）。
 * 10ms 程度以下の断片は processor が音声トークンを 1 つも作れず、音声エンコーダが失敗するか音声なしで生成が走るため。
 */
export const MIN_SEGMENT_SECONDS = 0.1;

export interface AudioSegment {
  /** 区間の開始位置（秒）。 */
  startSec: number;
  /** 区間の終了位置（秒）。 */
  endSec: number;
  /** 区間の波形。元配列の subarray なのでコピーしない。 */
  samples: Float32Array;
}

/** 各チャンネルを等しい重みで平均してモノラルにする。 */
export function downmixToMono(channels: readonly Float32Array[]): Float32Array {
  const [first] = channels;
  if (first === undefined) return new Float32Array(0);
  if (channels.length === 1) return first;

  const mono = new Float32Array(first.length);
  for (let i = 0; i < mono.length; i++) {
    let sum = 0;
    for (const channel of channels) sum += channel[i] ?? 0;
    mono[i] = sum / channels.length;
  }
  return mono;
}

/**
 * 波形を先頭から `maxSeconds` 秒ごとの区間に切る。最後の区間だけ短くなりうるが、`minSeconds` 未満なら捨てる。
 */
export function splitIntoSegments(
  audio: Float32Array,
  samplingRate = SAMPLING_RATE,
  maxSeconds = MAX_SEGMENT_SECONDS,
  minSeconds = MIN_SEGMENT_SECONDS,
): AudioSegment[] {
  const segmentLength = Math.floor(samplingRate * maxSeconds);
  const minLength = Math.ceil(samplingRate * minSeconds);
  const segments: AudioSegment[] = [];
  for (let start = 0; start < audio.length; start += segmentLength) {
    const end = Math.min(start + segmentLength, audio.length);
    const isTooShort = end - start < minLength;
    if (isTooShort) break;
    segments.push({
      startSec: start / samplingRate,
      endSec: end / samplingRate,
      samples: audio.subarray(start, end),
    });
  }
  return segments;
}

/** 秒数を四捨五入して `mm:ss` 形式にする（1 時間以上は分が 60 を超える）。 */
export function formatTimestamp(totalSeconds: number): string {
  // 切り捨てだと 8.98 秒の音声が 00:08 と表示され、状況表示の「9.0 秒」と食い違うため四捨五入する
  const seconds = Math.max(0, Math.round(totalSeconds));
  const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
  const ss = String(seconds % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

/**
 * ブラウザの音声デコーダで任意形式（webm/opus・mp3・m4a・wav など）を 16kHz モノラルに変換する。
 * AudioContext は Worker から使えないため、メインスレッドで呼ぶ。
 */
export async function decodeToMono16k(blob: Blob): Promise<Float32Array> {
  // OfflineAudioContext ではなく AudioContext を使うのは、decodeAudioData の再サンプリング先を sampleRate で指定できれば足り、長さを事前に知る必要がないため
  const context = new AudioContext({ sampleRate: SAMPLING_RATE });
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const channels = Array.from({ length: decoded.numberOfChannels }, (_, ch) => decoded.getChannelData(ch));
    return downmixToMono(channels);
  } finally {
    await context.close();
  }
}
