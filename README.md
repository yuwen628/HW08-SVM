# SVM 核技巧 3D 互動式展示 V2

## 線上 Demo

立即體驗 Streamlit 互動式展示：

<https://hw08-svm-xhiftjyx5hvfmncau93y9h.streamlit.app/>

## 專案簡介

本專案是一套適合課堂教學使用的支援向量機（Support Vector Machine, SVM）核技巧展示，內容包含：

- 使用 Manim 製作概念動畫，示範如何將環狀的二維資料映射至三維空間。
- 使用 sklearn 建立真實的 RBF SVM 決策函數視覺化。
- 使用 Streamlit 與 Plotly 製作可調整核函數參數的互動式應用程式。
- 以動態模型效果呈現決策曲面如何從較簡單的模型逐步變化至側邊欄所選的參數設定。

## 教學情境

資料集中，藍色資料點分布在原點附近，紅色資料點則分布在外圍環帶。在原始二維平面上，無法使用一條直線將兩個類別分開；透過特徵映射，同一組資料可以在更高維度的空間中變成線性可分。

## 第一階段：Manim 核技巧動畫

檔案：`phase1_manim_kernel_trick.py`

動畫使用以下教學用映射：

```text
phi(x, y) = (x, y, x^2 + y^2)
```

內圈的藍色資料點維持在較低位置，外圈的紅色資料點被提升至較高位置，再以一個水平平面在三維空間中將它們分開。將此平面投影回二維平面後，便會形成圓形的決策邊界。

## 第二階段：真實 RBF SVM 決策曲面

檔案：`phase2_rbf_decision_surface.py`

此程式使用 sklearn 的 `SVC(kernel="rbf")` 訓練真實模型，並繪製：

- 二維決策邊界 `f(x, y) = 0`
- 間隔等高線 `f(x, y) = -1` 與 `f(x, y) = +1`
- 三維決策曲面 `z = f(x, y)`

支援向量會在二維與三維圖表中以醒目樣式標示。

## 第三階段：Streamlit 互動式展示

檔案：`phase3_streamlit_app.py`

使用者可以調整以下參數：

- 核函數：`linear`、`poly`、`rbf`、`sigmoid`
- `C`
- `gamma`
- 多項式次數 `degree`
- 雜訊程度
- 資料點數量
- 隨機種子
- 網格解析度

調整參數後，二維決策邊界、三維決策函數曲面、支援向量數量與教學說明都會即時更新。

V2 新增「動態模型效果」區塊。程式會逐步增加 `C`，並針對非線性核函數逐步增加 `gamma`，建立數個中間 SVM 模型，再透過 Plotly 動畫控制項呈現三維決策函數曲面的變化。

## 安裝方式

安裝第二、三階段所需套件：

```bash
pip install -r requirements.txt
```

如需執行第一階段的動畫，請另外安裝 Manim：

```bash
pip install manim
```

若在 Windows 安裝 Manim 時遇到 `moderngl` 或 `glcontext` 編譯錯誤，建議使用 Python 3.11 或 3.12 建立 Manim 環境，或先安裝 Microsoft C++ Build Tools 再重新執行安裝。sklearn、Matplotlib、Streamlit 與 Plotly 等功能不依賴 Manim。

## 執行方式

請先切換至 `HW08-SVM-v2` 目錄，再依需求執行：

```bash
# Manim 低畫質預覽
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D

# Manim 高畫質輸出
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D

# RBF SVM 決策曲面
python phase2_rbf_decision_surface.py

# Streamlit 互動式應用程式
streamlit run phase3_streamlit_app.py
```

## 重要數學說明

映射 `z = x^2 + y^2` 是用於視覺化與教學的特徵映射，目的是說明非線性資料為何能在高維特徵空間中變成線性可分。

真實的 RBF 核函數並不會明確地將資料映射至三維空間，而是對應至高維甚至無限維的特徵空間。因此，第二與第三階段所呈現的 RBF 決策曲面是決策函數 `f(x, y)` 的視覺化，而不是完整特徵空間本身。

## 教學建議

- 從第一階段開始，建立核技巧的幾何直覺。
- 使用第二階段說明 sklearn 如何訓練真實的 RBF SVM。
- 在課堂上操作第三階段，請學生預測調整 `C` 或 `gamma` 後的變化。
- 比較較小與較大的 `gamma`，討論決策邊界的平滑程度與過度擬合。
- 比較較小與較大的 `C`，討論軟間隔的效果。
