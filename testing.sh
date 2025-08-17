#!/bin/bash
export SUMO_HOME=~/sumo-1.9.2/
export PATH=$SUMO_HOME/bin:$PATH
source ~/blockchain/bin/activate
python3 main.py --base-dir real_a1/ma2c_nc evaluate
source ~/blockchain/bin/activate
python3 main.py --base-dir real_a1/ma2c_nclm evaluate
source ~/blockchain/bin/activate
python3 main.py --base-dir real_a9/ma2c_nc evaluate
source ~/blockchain/bin/activate
python3 main.py --base-dir real_a9/ma2c_nclm evaluate
source ~/blockchain/bin/activate
python3 main.py --base-dir training/real/base/ma2c_nclm/ --port 100 evaluate
python3 main.py --base-dir training/real/direction_aw001c10/ma2c_nclm --port 135 evaluate