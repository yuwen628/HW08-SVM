# L13-SVM 工作紀錄

日期：2026-06-18

## 執行依據

- 先讀取 `CLAUDE.md`，採用其規範：
  - 先確認假設與成功條件。
  - 保持實作簡潔。
  - 只建立設計檔要求的必要檔案。
  - 完成後以可驗證方式檢查。
- 再讀取 `L13-2/design.md`，依其內容建立 SVM Kernel Trick 3D demo 專案。

## 完成內容

建立完整專案結構與交付檔案：

- `requirements.txt`
- `README.md`
- `phase1_manim_kernel_trick.py`
- `phase2_rbf_decision_surface.py`
- `phase3_streamlit_app.py`
- `utils/__init__.py`
- `utils/data_generator.py`
- `utils/svm_utils.py`
- `assets/`
- `outputs/`

## 功能摘要

### Phase 1: Manim 動畫

- 建立 `SVMKernelTrick3D` scene。
- 使用教學用映射：

```text
phi(x, y) = (x, y, x^2 + y^2)
```

- 展示 2D 圓形資料無法線性分割。
- 將點提升到 3D。
- 顯示拋物面、水平 hyperplane，以及投影回 2D 的圓形 decision boundary。
- README 中明確註記：這是教學用 3D 映射，不是宣稱 RBF kernel 只映射到 3D。

### Phase 2: 真實 RBF SVM 視覺化

- 使用 sklearn `SVC(kernel="rbf")` 訓練模型。
- 顯示 2D decision boundary `f(x, y) = 0`。
- 顯示 margin contours `f(x, y) = -1` 與 `f(x, y) = +1`。
- 顯示 3D decision function surface `z = f(x, y)`。
- 標記 support vectors。
- 輸出成果圖：`outputs/rbf_decision_surface.png`。

### Phase 3: Streamlit/Plotly 互動展示

- 建立互動 app：`phase3_streamlit_app.py`。
- 支援調整：
  - kernel
  - C
  - gamma
  - degree
  - noise
  - number_of_points
  - random_seed
  - grid_resolution
- 顯示：
  - 2D decision boundary
  - 3D decision function surface
  - support vector 數量
  - training accuracy
  - 動態 teaching notes

## 驗證紀錄

已執行：

```bash
python -m compileall .
```

結果：Python 語法編譯通過。

已測試 sklearn 共用工具：

- dataset generation 成功。
- SVM training 成功。
- decision grid 建立成功。
- decision surface 計算成功。

已測試 Phase 2：

- matplotlib 使用 non-interactive backend 執行成功。
- 成功產生 `outputs/rbf_decision_surface.png`。

已測試 Phase 3：

- Plotly 2D figure 建構成功。
- Plotly 3D figure 建構成功。
- RBF 與 poly kernel 的圖形建構測試成功。
- Streamlit server 啟動成功。

啟動網址：

```text
http://localhost:8501
```

HTTP 回應狀態：`200`

## 環境限制

嘗試安裝完整 `requirements.txt` 時，Manim 的相依套件 `moderngl` / `glcontext` 在目前 Windows + Python 3.14 環境需要 Microsoft C++ Build Tools，因此 Manim 未完成安裝。

處理方式：

- 已先安裝並驗證 Phase 2/3 所需套件：
  - `scikit-learn`
  - `matplotlib`
  - `plotly`
- README 已加入 Manim 安裝提醒：
  - 可改用 Python 3.11 或 3.12 建立 Manim 環境。
  - 或安裝 Microsoft C++ Build Tools 後重新執行 `pip install -r requirements.txt`。

## 最終狀態

- Phase 2/3 可執行並已驗證。
- Streamlit app 已啟動於 `http://localhost:8501`。
- Manim 腳本已完成，但受本機 Manim 安裝環境限制，尚未進行實際 render。

## V2 update - 2026-06-18

- Copied `WH8-SVM` to `WH8-SVM-v2`; original project is unchanged.
- Added a Streamlit **Dynamic Model Effect** section.
- The dynamic view builds intermediate SVM models and animates the 3D decision
  surface from lower `C` / `gamma` values toward the selected sidebar settings.
- Updated README title, V2 description, and run directory.

## V2 dynamic gamma update - 2026-06-18

- Changed the Dynamic Model Effect gamma progression from 6 stages to 15 stages.
- Added `DYNAMIC_GAMMA_STEPS = 15` in `phase3_streamlit_app.py`.
- Verified the dynamic surface generator now returns 15 animation frames.

## V2 header image update - 2026-06-18

- Generated a technology-themed SVM header image with the built-in `image_gen`
  tool.
- Saved the project asset as `assets/svm-tech-header.png`.
- Added the header image above the Streamlit title in `phase3_streamlit_app.py`.
- The image is displayed at full container width with a fixed height of 300px.
- Verified `phase3_streamlit_app.py` compiles successfully.

## V2 language toggle update - 2026-06-18

- Added an English / Chinese toggle at the top of the Streamlit sidebar.
- Added a `TEXT` dictionary in `phase3_streamlit_app.py` for bilingual labels
  and teaching notes.
- Updated the page title, sidebar controls, metrics, plot labels, Dynamic Model
  Effect play button, and teaching notes to switch language from the toggle.
- Verified the English and Chinese text paths and dynamic figure generation.
