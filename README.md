<img width="1312" height="1144" alt="LocalAIMeetingTool-v4 5_SS" src="https://github.com/user-attachments/assets/407193ab-4b34-4230-9648-ccc6b2afcd12" />

# Local AI Meetingtool

ローカル実行を前提にした、音声・動画の文字起こし、話者分離、要約アプリです。

macOS版はApple Silicon向けに `mlx-whisper` を使います。Windows版はWindows専用として `faster-whisper` を使います。要約はLM StudioまたはOllama CLIのローカルLLMを利用できます。

## Versions

- `Local_AI_Meetingtool_mac-v4.py`
  - Apple Silicon Mac向けの通常版です。
  - M2 / 16GB環境で動作確認しています。

- `Local_AI_Meetingtool_Pro_mac-v4.5.py`
  - Apple Silicon Mac向けのPro版です。
  - M5 Pro / 48GB環境で動作確認しています。
  - Large v3が安定動作するようにチューニングしています。

- `Local_AI_Meetingtool_Pro_Windows-v4.5.py`
  - Windows専用のPro版です。
  - `faster-whisper` を使用します。
  - 動作確認環境: AMD Ryzen 7 7800X3D / RAM 32GB / RTX 5070 Ti 16GB

## Features

- 日本語音声・動画の文字起こし
- `pyannote-audio` による話者分離
- LM StudioまたはOllama CLIを使ったローカルLLM要約
- 専門用語・略語の登録
- 話者名の置き換え
- 個人情報のマスキング
- テキスト形式とWord形式での保存
- Hugging FaceトークンのOS資格情報ストア保存

## Requirements

macOS版:

- macOS
- Apple Silicon Mac
- Python 3.14
- Homebrew
- ffmpeg
- LM Studio、またはOllama CLI
- `requirements-mac.txt`

Windows版:

- Windows 10 / 11
- Python 3.14
- ffmpeg
- LM Studio、またはOllama CLI
- CUDA対応GPU推奨
- `requirements-windows.txt`

## Setup

macOS:

```bash
brew install python-tk@3.14 ffmpeg
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-mac.txt
```

Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

ffmpegは別途インストールし、`ffmpeg` コマンドをPATHから実行できる状態にしてください。

## Run

macOS通常版:

```bash
.venv/bin/python Local_AI_Meetingtool_mac-v4.py
```

macOS Pro版:

```bash
.venv/bin/python Local_AI_Meetingtool_Pro_mac-v4.5.py
```

Windows Pro版:

```powershell
.\.venv\Scripts\python.exe .\Local_AI_Meetingtool_Pro_Windows-v4.5.py
```

## Hugging Face

話者分離を使う場合は、Hugging Faceで以下のモデルへのアクセス権が必要です。

- `pyannote/speaker-diarization-3.1`

トークンはアプリ画面に入力できます。入力したトークンは設定JSONには保存せず、`keyring` 経由でOS資格情報ストアへ保存します。

すでに `huggingface-cli login` などでローカル認証済みの場合は、トークン欄を空のままでも動作します。
※トークン欄に何か入力されていると、トークンと誤認するので注意してください。

## Summary Backends

LM Studioを使う場合:

- LM Studioでローカルサーバーを起動してください。
- OpenAI互換APIを有効にしてください。
- URLは `http://localhost:1234/v1` です。
- 要約用モデルをロードしてから実行してください。

Ollama CLIを使う場合:

- `ollama` コマンドをPATHから実行できる状態にしてください。
- アプリからOllamaモデル名を指定できます。
- Ollamaサーバーが未起動の場合、アプリが `ollama serve` の起動を試みます。

## Configuration

設定ファイル:

- macOS版設定: `whisper_config.json`
- Windows版設定: `whisper_config_windows.json`

Windows版では `whisper_config_windows.json` を使用します。処理性能に関わる値はこのファイルで調整できます。

例:

```json
{
  "batch_size": 192,
  "context_length": 8000,
  "windows_whisper_device": "cuda",
  "windows_whisper_compute_type": "float16",
  "windows_whisper_cpu_threads": 8,
  "windows_whisper_num_workers": 1,
  "windows_whisper_beam_size": 5,
  "windows_whisper_best_of": 3,
  "windows_whisper_batch_size": 16,
  "windows_whisper_use_batched": true
}
```

- `batch_size`: 話者分離(pyannote)のバッチサイズです。メモリ不足時は小さくしてください。
- `context_length`: 要約時にローカルLLMへ渡す1チャンクあたりの目安文字数です。
- `windows_whisper_device`: `cuda` または `cpu` を指定します。
- `windows_whisper_compute_type`: CUDA使用時の計算型です。
- `windows_whisper_batch_size`: Windows版のbatched inferenceのバッチサイズです。
- CUDAが検出できない場合やCUDA実行時に失敗した場合、Windows版のWhisperは自動的にCPU `int8` へフォールバックします。

## Local Files

以下はローカル設定・生成物として扱い、Gitには含めません。

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- `app.log`
- `whisper_config.json`
- `whisper_config_windows.json`
- `*_transcript_*.txt`
- `*_summary_*.txt`
- `*.docx`

## License

MIT License. See [LICENSE](LICENSE).
