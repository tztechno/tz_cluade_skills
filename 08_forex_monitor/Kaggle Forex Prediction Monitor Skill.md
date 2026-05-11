  
  
## Kaggle Forex Prediction Monitor Skill  
Kaggle Notebook上でドル円（USD/JPY）の1分足データを取得し、実データと未来予測をインタラクティブに可視化・更新し続けるスキル。  
## 構成  
Kaggle Notebook（Python）単一のセル、または一連のセルで完結する構成とする。  
## 機能要件  
## 1. データ取得  
* **yfinance** ライブラリを使用（Kaggle環境で標準利用可能かつ認証不要）。  
* USDJPY=X の **1分足** データを取得。  
* 初回：過去120分（2時間）分を取得。  
* ループ：1分ごとに最新の1分足を追加取得。  
## 2. 予測機能  
* 過去データ（120本）を基に、未来60分（1時間）を予測。  
* アルゴリズム：**statsmodels.tsa.arima.model.ARIMA** または **ExponentialSmoothing** を使用。  
* 実行速度を優先し、次数選択は簡易的なもの（(5,1,0)等）で固定可。  
## 3. Notebook ダッシュボード  
* **Plotly (Graph Objects)** を使用してインタラクティブなグラフを生成。  
* **IPython.display.clear_output** を使用し、1分ごとにグラフを再描画（Live Monitor形式）。  
* **表示内容**：  
    * 青実線：実データ（最新120分）  
    * 赤破線：予測データ（未来60分）  
    * 垂直線：現在時刻（予測の開始点）  
    * 直近の予測値と実値の乖離（MAE/RMSE）をサブタイトルまたはアノテーションで表示。  
## 4. 継続的な更新ループ  
* while True ループによる1分ごとのデータ取得・再予測・描画。  
* 1分間の待機（time.sleep(60)）を挟む。  
* Kaggleのセッション実行時間制限に配慮し、手動停止（KeyboardInterrupt）でクリーンに終了する構造。  
## 技術スタック  
* **Language**: Python 3 (Kaggle Notebook)  
* **Data Source**: yfinance  
* **Forecasting**: statsmodels, numpy  
* **Visualization**: Plotly  
## 実装イメージ (Kaggle Notebook 構成)  
Python  
##   
##   
##   
##   
##   
##   
## import yfinance as yf  
## import pandas as pd  
## from statsmodels.tsa.arima.model import ARIMA  
## import plotly.graph_objects as go  
## from IPython.display import clear_output  
## import time  
##   
**# 1. 初期データ取得**  
**# 2. 無限ループ:**  
**#    a. 最新1分足の取得 & データのスライド**  
**#    b. ARIMAモデルによる予測**  
**#    c. Plotlyによる描画**  
**#    d. clear_output(wait=True) で画面更新**  
**#    e. time.sleep(60)**  
## 注意事項  
* **APIレート制限**: yfinance (Yahoo Finance) の短時間での過度なリクエストを避けるため、スリープ時間は厳守すること。  
* **メモリ管理**: ループ内で古いデータを切り捨て、メモリの増大を防ぐこと。  
* **UIメッセージ**: 予測は統計モデルに基づくものであり、投資助言ではない旨をグラフ内に明記すること。  
  
## 変更のポイント  
1. **バックエンド/フロントエンドの統合**: Kaggle Notebook上ではサーバーを立てる必要がない（かつポート開放が難しいため）、IPython.display を使ったループ更新型に変更しました。  
2. **APIの選定**: Twelve Data（要APIキー）から、Kaggleで最も手軽に使える yfinance に変更しました。  
3. **ライブラリ**: requirements.txt を別途用意する代わりに、Kaggleの標準プリインストールライブラリで完結させています。  
  
