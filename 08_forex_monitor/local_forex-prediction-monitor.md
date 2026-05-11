  
name: forex-prediction-monitor  
description: ドル円の過去2時間の1分足データを取得し、実データグラフと未来1時間の予測グラフを表示。その後継続的に1時間分の実データを追加取得し、予測と比較・更新するWebダッシュボードを生成する。  
---  
  
# Forex Prediction Monitor  
  
為替APIからドル円の1分足データを取得し、実データと予測を可視化・比較監視するダッシュボードを生成するスキル。  
  
## 出力成果物  
  
以下のファイルを生成すること：  
  
- `app.py` - FastAPIまたはFlaskベースのバックエンド  
- `templates/index.html` - グラフ表示用フロントエンド  
- `requirements.txt` - 依存ライブラリ一覧  
- `README.md` - 実行手順  
  
## 機能要件（実装必須）  
  
### 1. データ取得  
- 無料の為替APIを使用する（候補：Twelve Data、Forex Open API、またはYahoo Finance非公式API）  
- ドル円（USD/JPY）の**1分足**データを取得すること  
- 初回：**過去2時間分（120本）** を取得  
- 継続的：1分ごとに新しい**実データ**を追加取得  
  
### 2. 予測機能  
- 過去2時間の実データを基に、**未来1時間（60分分）** のレートを予測すること  
- 予測アルゴリズム：簡易的なもので可（移動平均ベースのトレンド extrapolation、またはProphet／ARIMA）  
- 予測値は60個（1分刻み）生成すること  
  
### 3. Webダッシュボード  
- ブラウザで表示可能なHTMLを生成すること  
- グラフライブラリ：Plotly.js または Chart.js を使用  
- **表示内容**：  
  - 実データ（過去2時間 → 青線）  
  - 予測データ（未来1時間 → 赤破線）  
  - X軸：3時間分（過去2h＋未来1h）の時間軸  
  - 現在時刻を示す縦の点線  
  
### 4. 継続的な比較更新  
- 初回表示後もバックグラウンドで動作すること  
- **1分ごと**に新しい実データを取得し、グラフに追加反映すること  
- 過去2時間のウィンドウは最新データに合わせて**スライド**すること（古いデータは捨てる）  
- 予測値と実データが同じ時間軸上で**比較できること**  
- オプション：予測誤差（MAEなど）を画面に表示する  
  
### 5. APIエンドポイント（バックエンド設計）  
以下のエンドポイントを実装すること：  
  
| エンドポイント | 役割 |  
|--------------|------|  
| `/` | ダッシュボードHTMLを返す |  
| `/api/initial` | 初回データ（過去2h実データ + 未来1h予測）をJSONで返す |  
| `/api/latest` | 最新の1分実データを返す（更新用） |  
  
## 技術スタック（指定）  
  
- バックエンド：Python + FastAPI または Flask  
- フロントエンド：HTML + JavaScript + Plotly.js  
- データ保存：メモリ（Pythonの変数／リスト）で十分  
- 予測：statsmodels（ARIMA）または numpy.polyfit による単純回帰  
  
## ファイル構成例  
  
```  
project/  
├── app.py  
├── templates/  
│   └── index.html  
├── static/  
│   └── (必要なし、PlotlyはCDN使用)  
├── requirements.txt  
└── README.md  
```  
  
## エラーハンドリング（実装必須ではないが推奨）  
  
- API取得失敗時のリトライ機構  
- データ欠損時の補間処理  
- 予測モデル計算不能時のフォールバック（直前値で補完）  
  
## 注意事項  
  
- APIキーが必要な場合は、環境変数から読み込むこと（コード内ハードコード禁止）  
- 無料APIのレート制限に注意すること（キャッシュ・更新間隔調整）  
- 予測はあくまで参考値であり、正確を保証しない旨をUIに明記すること  
  
## 実行確認  
  
生成後、以下のコマンドで動作すること：  
  
```bash  
pip install -r requirements.txt  
python app.py  
# → http://localhost:8000 でダッシュボード表示  
```  
  
## 補足：使用するAPIの具体例  
  
**推奨：Twelve Data**（無料枠あり、1分足対応）  
- エンドポイント: `https://api.twelvedata.com/time_series`  
- パラメータ: `symbol=USD/JPY`, `interval=1min`, `outputsize=120`  
  
または **Yahoo Finance**（非公開API、認証不要）  
- `https://query1.finance.yahoo.com/v7/finance/download/USDJPY=X`  
  
