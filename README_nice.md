
**完全WORK**
**目標：** 讓 Docker 容器在背景穩定運行，獨立於您的 SSH 連線，執行想要的RL訓練。

**前提：**

* 您已經在主機上安裝好 Docker 和 NVIDIA Docker 支持。
* 您的 `Dockerfile` 位於 `~/best_environment/` 目錄下。
* 您的專案程式碼位於主機的 `/home/russell512/my_deeprl_network_ori_test/`。
* 上面都沒問題，就是跑不同的資料夾中的程式碼，可能要改一下資料夾名稱這樣
---

**一步步指南：**

**第 1 步：建置 Docker Image (與您原本流程相同)**

```bash
cd ~/best_environment
docker build -t best_environment:latest .
```

**第 2 步：啟動 Docker 容器於背景 (Detached 模式 - 關鍵改變)**

* 這一步是核心改變，我們讓容器在背景啟動並保持運行。
* 打開您的主機終端機 (例如透過 SSH 登入後)。
* 執行以下指令：

    ```bash
    docker run \
      --gpus all \
      -d \
      --name Russell_Trainer_0417 \
      -v /home/russell512/my_deeprl_network_ori_test:/workspace/my_deeprl_network \
      best_environment:latest \
      sleep infinity
    ```
    * 上面都沒問題，就是跑不同的資料夾(/home/russell512/my_deeprl_network_ori_test)中的程式碼，可能要改一下-v的資料夾名稱這樣
    * `-d`: Detached 模式，讓容器在背景運行。 
    * `--name Russell_Trainer_0417`: 給容器取一個固定且易於識別的名稱。
    **請確保這個名稱是唯一的****請確保這個名稱是唯一的****請確保這個名稱是唯一的****請確保這個名稱是唯一的****請確保這個名稱是唯一的**
    
    如果已存在，請換一個或先用 `docker rm Russell_Trainer_0417` 刪除舊的。
    * `sleep infinity`: 讓容器啟動後執行一個永遠不會結束的命令，以保持容器持續運行。

**第 3 步：進入正在運行的容器**

* 容器已經在背景運行了，現在我們需要進入它來執行設定和啟動訓練。
* 執行：

    ```bash
    docker exec -it Russell_Trainer_0417 /bin/bash
    ```
    * 您現在應該會看到容器內部的命令提示符 (類似 `root@<container_id>:/#`)。

**第 4 步：在容器內部進行環境設置 (與您原本流程相同)**

* **您現在是在容器裡面操作。**
* 執行您原本就需要做的設定：

    ```bash
    # 安裝 pip 套件
    pip install traci
    pip install sumolib
    pip install torch

    # 進入工作目錄
    cd /workspace/my_deeprl_network

    # 設定環境變數
    export SUMO_HOME="/root/miniconda/envs/py36/share/sumo" 

    # 安裝 tmux
    apt update
    apt install -y tmux
    ```

**第 5 步：在容器內部，使用 tmux 啟動訓練 (與您原本流程相同)**

* **您仍然在容器裡面操作。**
* 啟動一個新的 `tmux` 會話（建議使用新名稱以便區分）：
* 可以改名稱
    ```bash
    tmux new -s training_session_0417
    ```
* **在新的 `tmux` 視窗中**，執行您的訓練指令：
*全部輸出都到log

export USE_GAT=1
python3 test.py \
  --base-dir real_a1/dp_decay \
  --port 195 \
  train \
  --config-dir config/config_ma2c_nc_net_ten_times.ini \
  > real_a1/dp_decay/log/training_gat1_dp_decay_$(date +%Y%m%d_%H%M%S).log 2>&1

    ```
* 確認訓練腳本已經開始運行並輸出日誌。

**第 6 步：從 tmux 會話分離 (與您原本流程相同)**

* 按下 `Ctrl + b` 組合鍵，放開，然後再按 `d` 鍵。
* 您會看到 `[detached (from session training_session_0417)]` 訊息，並回到容器的命令提示符 (`root@...`)。

**第 7 步：退出容器的 exec 連線**

* 在容器的命令提示符下，輸入 `exit` 並按 Enter。
* 您會回到主機的命令提示符 (`(base) $` 或其他)。

**第 8 步：斷開 SSH 連線**

* **現在您可以安全地關閉 SSH 連線或 VS Code 了。**
* 由於容器是以 `-d` 模式和 `sleep infinity` 啟動的，它會一直在背景運行，容器內的 `tmux` 會話 (`training_session_0417`) 和 Python 腳本也會繼續執行。

**第 9 步：如何重新連線查看進度**

1.  重新透過 SSH 登入您的主機。
2.  執行 `docker exec -it Russell_Trainer_0417 /bin/bash` 再次進入容器。
3.  在容器內，執行 `tmux attach -t training_session_0417` 重新連接到您的 `tmux` 會話查看訓練情況。
4.  查看完畢後，可以再次用 `Ctrl+b`, `d` 分離，然後 `exit` 退出容器。

---

這個流程的核心改動在於**第 2 步**，確保了容器的獨立運行。其他在容器內部的操作步驟與您原本的習慣幾乎完全一致。遵循這個流程，可以最大程度避免因斷開連線導致容器停止的問題，讓您的訓練持續進行（只要訓練本身不報錯且資源充足）。