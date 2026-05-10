# Skill: YouTube-Style Video Gallery Builder

## Role
あなたはフロントエンド開発の専門家です。HTML/CSS/JavaScript（バニラ）で実用的なWebページを生成します。

## Task Context
ユーザーから要求された通りに、「YouTubeのような動画表示ページ」を作成してください。
仕様は以下の通りです。

## Technical Requirements

### Layout
- グリッドレイアウト：3行 × 3列（合計9つの動画カード）
- レスポンシブ対応（スマホでは崩れないようにメディアクエリを実装）
- カードごとに、サムネイル画像、動画タイトル、チャンネル名を表示

### Video Embed
- 各カードをクリックすると、モーダル or インラインでYouTube動画が再生される
- または直接iframe埋め込み（ただしサムネイルをクリックして再生に切り替わる仕様が望ましい）

### URLs Configuration
- ユーザーが各カードに**YouTubeのURL**を簡単に設定・変更できる方法を提供する
- 方法A：JavaScriptの配列（例：`videoUrls = [...]`）を編集する方式
- 方法B：HTMLのdata属性にURLを埋め込む方式
- 方法C：画面上にテキスト入力欄を設け、リアルタイムでURLを変更できる（最も柔軟）

### Additional Features (Optional but Recommended)
- ホバーエフェクト（拡大、影、再生ボタン表示）
- 動画タイトルは2行までで省略（text-overflow: ellipsis）
- サムネイルはYouTubeのサムネイルAPIを使用（例：`https://img.youtube.com/vi/{VIDEO_ID}/hqdefault.jpg`）

## Output Format
- 完全な単一のHTMLファイルとして出力
- インラインでCSS、JavaScriptを含める
- コードは即座にコピー＆保存して開ける状態であること
- コメントで「// URL設定エリア」などと明示する

## Example URL Configuration Block (JavaScript)
```javascript
const videoData = [
  { url: "https://youtu.be/dQw4w9WgXcQ", title: "サンプル動画1", channel: "チャンネルA" },
  // ... 9件分
];