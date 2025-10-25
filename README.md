# Gemini Computer Use - Playwright版 最小実装

このプロジェクトは、[Google Gemini Computer Use Preview](https://github.com/google-gemini/computer-use-preview)をベースに、Playwrightのみを使用したシンプルな実装です。

## 主な特徴

### 1. Playwrightに特化
- 元のサンプルからSeleniumとAppiumの実装を削除し、Playwrightのみに特化
- よりシンプルで理解しやすいコードベース
- Playwrightの強力な機能（自動待機、リトライ機能など）を活用

### 2. 既存ブラウザへの接続対応
`PlaywrightComputer`クラスに`remote_debugging_port`引数を追加し、既存のブラウザインスタンスに接続可能になりました。

#### 既存ブラウザの起動方法

**Chrome/Chromiumの場合:**
```bash
# Mac
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222
```

**Microsoft Edgeの場合:**
```bash
# Mac
/Applications/Microsoft\ Edge.app/Contents/MacOS/Microsoft\ Edge --remote-debugging-port=9222

# Windows
"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9222

# Linux
microsoft-edge --remote-debugging-port=9222
```

### 3. 構造化出力とテキスト出力のサポート
エージェントループの終了時に`finalize_task`関数を実行するように変更し、以下の2つの出力形式に対応：

- **構造化出力**: `response_schema`パラメータにPydanticモデルを指定することで、タスク結果を構造化された形式で取得
- **テキスト出力**: `response_schema`を指定しない場合は、タスク結果の要約テキストを取得

### 4. トークン使用量の監視と制限
- 各iterationでトークン使用量を表示
- 最大入力/出力トークン数を設定可能
- 制限を超えた場合は自動的にループを終了

## インストール

```bash
# 依存関係のインストール
pip install google-genai playwright pydantic termcolor rich

# Playwrightのブラウザをインストール（新規ブラウザを起動する場合のみ必要）
playwright install chromium
```

## 使用方法

### 基本的な使い方

`main.py`ファイルを編集して、以下の部分を変更してください：

```python
if __name__ == "__main__":
    # タスクを指定
    query = "googleで東京の天気を検索して何度か教えて"

    # Gemini APIキーを設定
    api_key = "YOUR_API_KEY_HERE"  # あなたのAPIキーを入力

    # 基本的な実行
    main(query=query, api_key=api_key)
```

### 高度な使い方

#### 1. 構造化出力を使用する場合

```python
from pydantic import BaseModel
from typing import Optional

# 天気情報のスキーマを定義
class WeatherInfo(BaseModel):
    location: str
    temperature: Optional[str] = None
    weather_condition: Optional[str] = None
    additional_info: Optional[str] = None

# 構造化出力付きで実行
main(
    query="googleで東京の天気を検索して何度か教えて",
    api_key=api_key,
    response_schema=WeatherInfo  # 構造化出力のスキーマを指定
)
```

#### 2. トークン制限を設定する場合

```python
main(
    query=query,
    api_key=api_key,
    max_input_tokens=10000,   # 最大入力トークン数
    max_output_tokens=5000     # 最大出力トークン数
)
```

#### 3. 既存のブラウザに接続する場合

まず、リモートデバッグポートでブラウザを起動（上記参照）してから：

```python
def main(query: str, api_key: str, ...):
    env = PlaywrightComputer(
        screen_size=PLAYWRIGHT_SCREEN_SIZE,
        initial_url="https://www.google.com",
        highlight_mouse=True,
        close_on_exit=False,
        remote_debugging_port=9222  # 既存ブラウザのポート番号
    )
    # 以降は同じ
```

現在の`main.py`ではデフォルトで`close_on_exit=False`が設定されています。

#### 4. 複雑なタスクの例

```python
# 複数のステップを含むタスク
query = """
1. googleでPython公式サイトを検索
2. 最新バージョンの情報を確認
3. ダウンロードページに移動
"""

# タスク結果のスキーマ
class PythonVersionInfo(BaseModel):
    latest_version: str
    release_date: Optional[str] = None
    download_url: Optional[str] = None
    features: Optional[list[str]] = None

main(
    query=query,
    api_key=api_key,
    response_schema=PythonVersionInfo,
    max_input_tokens=15000,
    max_output_tokens=10000
)
```

### 実行

```bash
python main.py
```

## ファイル構成

- `agent.py`: Gemini Computer Use エージェントのメインロジック
  - `BrowserAgent`クラス: エージェントループの管理
  - `finalize_task`関数: タスク完了時の処理（動的生成）
  - トークン管理、構造化出力のサポート

- `client.py`: Playwrightクライアントの実装
  - `PlaywrightComputer`クラス: ブラウザ操作の実装
  - `EnvState`クラス: 環境状態（スクリーンショット、URL）の管理

- `main.py`: エントリーポイント
  - パラメータの設定とエージェントの起動
  - 構造化出力のスキーマ定義例

## 主な変更点（オリジナルとの違い）

### 1. Playwrightのみに特化
- Selenium、Appiumの実装を削除
- コードベースを約70%削減し、保守性を向上

### 2. 既存ブラウザ接続機能（新機能）
```python
PlaywrightComputer(
    remote_debugging_port=9222  # 既存ブラウザに接続
)
```
- 開発中のデバッグが容易
- 手動操作との組み合わせが可能

### 3. 統一的なタスク完了処理（新機能）
- すべてのタスクが`finalize_task`関数で終了
- 構造化出力とテキスト出力の両方に対応
- モデルが明示的にタスク完了を決定

### 4. トークン管理機能（新機能）
- リアルタイムでトークン使用量を監視
- 上限設定による安全な実行
- コスト管理が容易

## トラブルシューティング

### ブラウザが起動しない / 接続できない

1. **新規ブラウザを起動する場合:**
   - Playwrightのブラウザがインストールされているか確認: `playwright install chromium`

2. **既存ブラウザに接続する場合:**
   - ブラウザが正しいポートで起動されているか確認
   - ファイアウォール設定を確認

### APIエラーが発生する
- APIキーが正しく設定されているか確認
- Gemini APIの利用制限に達していないか確認
- モデル名が正しいか確認: `gemini-2.5-computer-use-preview-10-2025`

### トークン制限エラー
- `max_input_tokens`と`max_output_tokens`の値を増やす
- タスクを小さく分割する
- スクリーンショットの保持数を調整（`MAX_RECENT_TURN_WITH_SCREENSHOTS`）

### finalize_task関数が呼ばれない
- タスクの指示が明確か確認
- より具体的な完了条件を指定
- `response_schema`を使用して期待する出力を明確化

## 環境変数

```bash
# Gemini APIキー（オプション）
export GEMINI_API_KEY="your-api-key-here"
```

## 制限事項

- 現在はブラウザ操作のみ対応（デスクトップ操作は未対応）
- 単一タブでの操作のみ（マルチタブは自動的に単一タブに変換）
- PDFやOfficeファイルの直接操作は未対応

## ライセンス

Apache License 2.0（Google オリジナルコードのライセンスに準拠）

## 参考リンク

- [Google Gemini Computer Use Preview](https://github.com/google-gemini/computer-use-preview)
- [Playwright Documentation](https://playwright.dev/python/)
- [Google Gemini API Documentation](https://ai.google.dev/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 貢献

バグレポートや機能要望は、GitHubのIssuesでお願いします。