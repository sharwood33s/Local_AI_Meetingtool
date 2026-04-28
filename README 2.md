# MLX Whisper Pro

`review_target.py` を仮想環境で実行するためのセットアップメモです。

## 前提

- macOS
- Python 3.12 か 3.13 系を推奨
- LM Studio をインストールし、ローカルサーバーを起動しておく
- `ffmpeg` をインストールしておく
- 話者分離を使う場合は Hugging Face 認証済みであること

補足:
- このアプリは Windows 非対応です
- `LM Studio` 本体や `ffmpeg` は仮想環境には入りません

## フォルダ構成

```text
your-app/
  review_target.py
  requirements.txt
  README.md
  .gitignore
  .venv/
```

## 新規セットアップ

プロジェクトフォルダへ移動します。

```bash
cd /path/to/your-app
```

仮想環境を作成します。

```bash
python3 -m venv .venv
```

仮想環境を有効化します。

```bash
source .venv/bin/activate
```

`pip` を更新します。

```bash
python -m pip install --upgrade pip setuptools wheel
```

依存パッケージをインストールします。

```bash
pip install -r requirements.txt
```

## 起動方法

仮想環境を有効化してから起動します。

```bash
source .venv/bin/activate
python review_target.py
```

## LM Studio の前提

LM Studio 側では次を満たしてください。

- ローカルサーバーを起動している
- アプリが `http://localhost:1234/v1` に接続できる
- 要約用モデルとして `llama` 系モデルをロードしている

## ffmpeg の確認

`ffmpeg` が入っているか確認します。

```bash
ffmpeg -version
```

`command not found` の場合は Homebrew なら次で入れられます。

```bash
brew install ffmpeg
```

## Hugging Face 話者分離

話者分離を使う場合は、事前に Hugging Face 側の認証が必要なことがあります。

トークンをこのアプリに入力するか、ローカル環境にログイン済みの状態で使ってください。

## 新しい Mac へ移行するとき

アプリ本体をコピーした後、移行先で次を実行します。

```bash
cd /path/to/your-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python review_target.py
```

## 開発時のメモ

- 依存を更新したら `requirements.txt` も更新する
- 設定ファイル `whisper_config.json` はローカル用として扱う
- 将来 `review_target.py` を `app.py` などへリネームした場合は README の起動コマンドも合わせて修正する
