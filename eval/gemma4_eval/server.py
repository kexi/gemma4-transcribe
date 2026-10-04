"""評価用の静的サーバ。1 つの localhost オリジンで `/` → dist-eval/、`/data/` → eval/data/ を返す。

Why not 2 つのサーバ（ページとデータで別ポート）: 別オリジンになると評価ページの fetch に CORS ヘッダが要る。
アプリ本体（GitHub Pages）は同一オリジンでしか動かないので、評価も同じ条件に揃える。

使い方（手動確認用）: uv run --project eval python -m gemma4_eval.server --port 8001
"""

from __future__ import annotations

import argparse
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from .log import log
from .paths import DATA_DIR, DIST_EVAL_DIR

DATA_PREFIX = "/data"


class EvalRequestHandler(SimpleHTTPRequestHandler):
    """`/data/` 配下だけ data_dir から、それ以外は directory（dist-eval）から返す。"""

    # Python の既定の対応表には .mjs / .wasm が無い版があり、
    # application/octet-stream で返すと module Worker や WebAssembly.instantiateStreaming が拒否する
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
        ".wav": "audio/wav",
    }

    def __init__(self, *args, data_dir: Path, **kwargs) -> None:
        self._data_dir = data_dir
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        path_only = path.split("?", 1)[0].split("#", 1)[0]
        is_data = path_only == DATA_PREFIX or path_only.startswith(DATA_PREFIX + "/")
        if not is_data:
            return super().translate_path(path)
        # 親クラスの translate_path は self.directory を基準に `..` を取り除いて解決するので、
        # 基準だけ差し替えて同じ正規化を通す（自前で連結するとディレクトリトラバーサルを許しかねない）
        rest = path[len(DATA_PREFIX) :] or "/"
        original = self.directory
        self.directory = str(self._data_dir)
        try:
            return super().translate_path(rest)
        finally:
            self.directory = original

    def end_headers(self) -> None:
        # 評価のたびにビルドし直すので、ブラウザに古い runner を使わせない。
        # モデルは Transformers.js が Cache API に保存するため、ここでの no-store の影響を受けない
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        # 既定のアクセスログ（1 リクエスト 1 行の非 JSON）は進捗ログに混ざって読みにくいので出さない。
        # エラーだけは log_error 経由で JSON にする
        pass

    def log_error(self, format: str, *args) -> None:
        # favicon はブラウザが勝手に取りに来るだけで、評価に関係しない 404 なので出さない
        is_favicon = self.path.startswith("/favicon.ico")
        if is_favicon:
            return
        log("http_error", path=self.path, message=format % args)


def start_server(
    dist_dir: Path = DIST_EVAL_DIR,
    data_dir: Path = DATA_DIR,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """別スレッドでサーバを起動して返す。port=0 なら空いているポートを OS に選ばせる（server.server_port）。"""
    handler = functools.partial(EvalRequestHandler, directory=str(dist_dir), data_dir=data_dir)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="eval-server", daemon=True)
    thread.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dist-dir", type=Path, default=DIST_EVAL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    server = start_server(args.dist_dir, args.data_dir, port=args.port)
    # WebGPU は secure context が必要。localhost は http でも secure context として扱われる
    log("server_started", url=f"http://localhost:{server.server_port}/eval.html")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
