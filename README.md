Local AI Meetingtool 文字起こし & 要約アプリ
Apple Silicon (M1/M2/M3) チップに最適化された、ローカル完結型の文字起こし・話者分離・要約アプリケーションです。
mlx-whisper を使用することで、Mac上で高速かつ高精度な音声処理を実現します。
v2が文字起こし話者分離機能。v2.5がv2にLMStidioに接続し、文章を要約する機能を追加したものです。
high-performanceは当方の環境がM5Pro/46GB環境に移行したため作成したものです。
high-performance-v2はhigh-performanceにマニュアル作成モードを追加したものです。

🚀 主な機能
高精度文字起こし: mlx-whisper（Turbo, Large v3等）による高速なテキスト化。
話者分離 (Diarization): pyannote-audio を活用し、「誰がいつ話したか」を識別。
AI要約: LM Studio と連携し、ローカルLLM（Llama等）を用いた議事録作成・要約。
専門用語登録: 誤字を防ぐためのカスタムプロンプト（固有名詞・略語）指定。
マルチ形式保存: 文字起こし結果をテキスト（.txt）または Word（.docx）形式で出力可能。

🖥️ 動作環境
OS: macOS (Apple Silicon 搭載モデル推奨)
※ Windows では動作しません。
Python: 3.14 以上
推奨ハードウェア: Apple M1/M2/M3 チップ以降（GPU/MPS 加速を使用するため）
開発環境はApple M2/16GBです。

📦 必要ライブラリのインストール
使用前に以下のライブラリをインストールしてください。

Bash
pip install mlx-whisper pyannote.audio customtkinter python-docx openai torch
Note: 話者分離機能を使用するには、Hugging Faceで pyannote/speaker-diarization-3.1 へのアクセス権取得とトークンの発行が必要です。

🛠️ 事前準備（要約機能を使用する場合）
要約機能には LM Studio を使用します。
LM Studio を起動。
任意の Llama モデル（例: Llama-3 等）をロード。
Local Server タブを選択し、サーバーを起動（デフォルトポート: 1234）。
本アプリの「LM Studioで要約」ボタンが動作するようになります。

📖 使い方
python main.py（本スクリプト）を実行します。
1. ファイルを選択: mp3, m4a, mp4 などの音声・動画ファイルを選択します。

2. 設定:
話者分離を行いたい場合は「有効」にし、Hugging Face トークンを入力します。
使用する Whisper モデルを選択します（M2 16GB環境なら Turbo または Large v3 がおすすめ）。
端末側にトークンがインポートされている場合は、空欄でも動作します。
何か文字が入力されていると、Hugging Faceトークンと誤認しますので、トークン以外は入力しないでください。

4. 専門用語（任意）: AI:人工知能, ML:機械学習 のように登録すると変換精度が向上します。
また、テキストファイルやcsv形式の専門用語辞書をあらかじめ作成しておき、読み込みを行うことも可能です。
ファイル名の指定はありません。
文字起こしを開始する: 処理が完了するとテキストエリアに結果が表示されます。
保存・要約: 必要に応じて Text/Word で保存、または LM Studio で要約を実行します。

5.LMStudioで要約する場合
要約の精度は、選択するLLMの性能に依存します。日本語の会議記録などを要約する場合は、日本語に強いモデルを推奨します。

1.LM Studioを起動し、左側のメニューから Search (虫眼鏡アイコン) を開きます。
検索バーに以下のようなキーワードを入力し、モデル（GGUF形式）をダウンロードします。

おすすめの日本語対応モデル例:
Llama-3-8B-Instruct (ELYZAなどの日本語ファインチューニング版)
Qwen2.5-7B-Instruct
Swallow (Mistralベースの日本語モデル)

2.モデルサイズの選び方:
メモリ(RAM)容量に合わせて「Quantization (量子化)」のレベルを選びます。
一般的な環境（RAM 16GB程度）であれば、Q4_K_M や Q5_K_M などの「4bit〜5bit量子化」モデルを選ぶと、高速かつ安定して動作します。

3. ローカルサーバーの起動（必須）
本アプリは、LM Studioが提供する「OpenAI互換のローカルAPI」を利用してテキストを送信します。
左側のメニューから 「Local Server (↔アイコン)」 を開きます。
画面上部のプルダウンから、先ほどダウンロードしたモデルをロードします。
右側の設定パネル（Server Settings）を確認します。
Port: 1234（デフォルトのまま）
Cross-Origin-Resource-Sharing (CORS): ON

4.【重要】コンテキスト長 (Context Length) の設定
長時間の文字起こしデータを要約する場合、テキスト量がモデルの処理上限を超えてしまうことがあります。
右側のパネル内にある Context Length（または Context Size）の値を、モデルの許容範囲内（例：8192 や 32768 など）で大きめに設定しておくことをお勧めします。
画面上部の 「Start Server」 ボタンをクリックします。

⚠️ 注意事項
初回起動時: 各モデル（Whisper, pyannote）のダウンロードが行われるため、時間がかかる場合があります。
メモリ消費: Large v3 モデルを使用する場合、メモリ(RAM)を多く消費します。動作が重い場合は Turbo や Small をお試しください。
LM Studio: サーバーが起動していない状態で要約ボタンを押すとエラーになります。

📄 ライセンス
MIT License (GitHubに公開する際は適宜ライセンスファイルを作成してください)

💡 開発者メモ (環境移行用)
Hugging Face のキャッシュディレクトリを変更している場合は、環境変数 HF_HOME を移行先でも設定すること。
customtkinter の外観モードは MacBook の視認性に合わせた light モードに固定。

UIサイズは MacBook Air M2 (13インチ) の画面解像度 1000x800 に最適化済み。
