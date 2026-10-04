/**
 * Cache API に保存されたモデルのファイルを数える。評価ページがそのページでモデルをダウンロードしたかを判断するため。
 *
 * Why not Transformers.js の進捗（load-progress の loadedBytes）で判断する: Transformers.js はキャッシュから読むときも
 * レスポンスを進捗付きで読み、同じ progress イベントをバイト数付きで送るため、ダウンロードとキャッシュ読み込みを区別できない。
 * 保存されたファイル数の増減なら、実際にネットワークから取得して保存したときだけ増える
 */

/** CacheStorage のうち使う部分だけ。vitest で偽物を渡せるようにするため。 */
export interface CacheStorageLike {
  keys(): Promise<string[]>;
  open(name: string): Promise<{ keys(): Promise<readonly { url: string }[]> }>;
}

/**
 * Transformers.js がモデルのファイルを保存するときのキー（`https://huggingface.co/<id>/resolve/<revision>/<file>`）に含まれる部分。
 * revision まで含めるのは、別の revision のキャッシュを「保存済み」と数えないため
 */
export function modelFileKeyPart(id: string, revision: string): string {
  return `/${id}/resolve/${encodeURIComponent(revision)}/`;
}

/**
 * すべてのキャッシュから、このモデル（id + revision）のファイルの件数を数える。数えられなければ null。
 * Why not Transformers.js の env.cacheKey（transformers-cache）だけを見る: その値を読むには評価ページのメインスレッドに
 * Transformers.js を丸ごと読み込むことになり、キャッシュ名を設定で変えたときにも黙って 0 件になるため
 */
export async function countCachedModelFiles(
  storage: CacheStorageLike | undefined,
  id: string,
  revision: string,
): Promise<number | null> {
  if (storage === undefined) return null;
  const keyPart = modelFileKeyPart(id, revision);
  try {
    let count = 0;
    for (const name of await storage.keys()) {
      const cache = await storage.open(name);
      const requests = await cache.keys();
      count += requests.filter((request) => request.url.includes(keyPart)).length;
    }
    return count;
  } catch (error) {
    // シークレットウィンドウや容量の制限で Cache API が拒否されることがある。評価自体は止めない
    console.warn('Cache API を読めませんでした', error);
    return null;
  }
}
