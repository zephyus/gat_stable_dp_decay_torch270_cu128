#!/bin/bash
export SUMO_HOME=~/sumo/
export PATH=$SUMO_HOME/bin:$PATH
source ~/blockchain/bin/activate
python3 main.py --base-dir real_a9/ma2c_nc/ train --config-dir config/config_ma2c_nc_net.ini --port 100 &\
python main.py --base-dir training/real/bonus_wait_x_queue_q/ma2c_nclm/ --port 100 train --config-dir config/config_ma2c_nclm_net.ini
