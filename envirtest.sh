#!/bin/bash
# export SUMO_HOME=/home/russell512/miniconda3/envs/russenv
# export PATH=$SUMO_HOME:$PATH 
pwd
echo ----- PATH
printenv PATH
echo -------
printenv SUMO_HOME
which sumo
source /home/russell512/miniconda3/etc/profile.d/conda.sh
conda activate russenv

# 確認conda環境是否開啟成功
if command -v conda &> /dev/null
then
    # 確認是否在conda環境中
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo "Conda environment is not activated."
    else
        echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
    fi
else
    echo "Conda is not installed or not in your PATH."
fi