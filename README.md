全面升級至現代化的環境：Python 3.11 + PyTorch 2.7.0。
沒有動到核心演算法、模型結構或數學邏輯，僅進行必要的相容性更新與穩定性修復。

模型結構（層數/尺寸/拓樸）：未改動。Policy / LstmPolicy / FPPolicy / NCMultiAgentPolicy 的子模組與 head 之間的連接、維度與初始化邏輯一致。

演算法與數學邏輯（policy / value / loss）：維持等價；訓練時的分佈與損失計算未改變。改動多為裝置管理（CPU/GPU）一致性、shape 安全性與日誌/診斷清理。
