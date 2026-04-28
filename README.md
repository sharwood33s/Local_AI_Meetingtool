# Local AI Meetingtool

Apple Silicon Mac 向けの、ローカル実行を前提にした文字起こし・話者分離・要約アプリです。
`GUI_Whisper-v3.py`はM2/16GB環境で動作確認をしたものです。
メモリ使用率が高い場合はモデルを小さいものに変更し使用してください。
`GUI_Whisper_high-performance-v3.5.py`はM5Pro/48GB環境で動作を確認したものです。
Large V3が安定動作するようにチューニングしています。
メモリが不足する場合は、Turbo等に変更し使用してください。


## Features

- `mlx-whisper` による日本語音声・動画の文字起こし
- `pyannote-audio` による話者分離
- LM Studio のOpenAI互換ローカルAPIを使った要約
- 専門用語・略語の登録
- テキスト形式とWord形式での保存
- Hugging FaceトークンのOS資格情報ストア保存

## Requirements

- macOS
- Apple Silicon Mac
- Python 3.14
- Homebrew
- LM Studio
- ffmpeg

Windowsは対象外です。

## Setup

Homebrewで、PythonのTkinterサポートとffmpegを入れます。

```bash
brew install python-tk@3.14 ffmpeg
```

仮想環境を作成し、依存関係を入れます。

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python GUI_Whisper_high-performance-v3.5.py
```

## Hugging Face

話者分離を使う場合は、Hugging Faceで以下のモデルへのアクセス権が必要です。

- `pyannote/speaker-diarization-3.1`

トークンはアプリ画面に入力できます。入力したトークンは `whisper_config.json` には保存せず、`keyring` 経由でmacOS KeychainなどのOS資格情報ストアへ保存します。

すでに `huggingface-cli login` などでローカル認証済みの場合は、トークン欄を空のままでも動作する場合があります。

## LM Studio

要約機能を使う場合は、LM Studioでローカルサーバーを起動してください。

- OpenAI互換API
- URL: `http://localhost:1234/v1`
- 要約用モデルをロード済み

コード内の `api_key="lm-studio"` はLM Studio向けのダミー値です。実際のOpenAI APIキーではありません。

## Local Files

以下はローカル設定・生成物として扱い、Gitには含めません。

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- `whisper_config.json`
- `*_transcript_*.txt`

## License

MIT License. See [LICENSE](LICENSE).
