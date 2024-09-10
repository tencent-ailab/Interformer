#!/bin/bash
sleep $((RANDOM % 10))
node_list=$NODE_LIST
gpu_num=$NODE_NUM
net_devices=mlx

echo $NODE_LIST
echo "GPU_NUM:$NODE_NUM"

pids=`ps -ef | grep "python train.py" | grep -v grep | awk '{print $2}'`
if [ "$pids" != "" ]
then
    echo $pids | xargs kill -9
fi
# check gpus
nvidia-smi


export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH > ld_library_paty.log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Freeing Memory"
echo 3 > /proc/sys/vm/drop_caches
echo "Current Path:"$(pwd)
######
SHARE_DIR=$(realpath /apdcephfs_qy3/share_*/revoli)
MODEL_DIR=$SHARE_DIR/inter
SAVE_DIR=/dockerdata
echo "TJ_INSTANCE_ID->${TJ_INSTANCE_ID}"
echo "SHARE_DIR->${SHARE_DIR}"
echo "MODEL_DIR->${MODEL_DIR}"
echo "TAIJI_JIZHI_WORKSPACE_PATH->${JIZHI_WORKSPACE_PATH}}"

ls -al
echo '*************Split********************'
# COPY EXTRA FOLDER TO LOCAL
cp -r $SHARE_DIR/extra $SAVE_DIR/
mkdir -p ${MODEL_DIR}/logs
echo '+++++++++++++Train++++++++++++++++++++'

# Start by single machine
my_port=2345
export NCCL_IB_DISABLE=1
export TORCHELASTIC_RESTART_COUNT=0
echo "# Checking Parameters, HOST_NUM=$HOST_NUM, MASTER_ADD=$CHIEF_IP, my_port=$my_port, HOST_GPU_NUM=$HOST_GPU_NUM, INDEX:$INDEX"
PYTHONPATH=interformer/ python -m torch.distributed.run --nnodes=$HOST_NUM --node_rank=$INDEX --nproc_per_node=$HOST_GPU_NUM --master_addr=$CHIEF_IP --master_port=$my_port --max_restarts 0 train.py \
-num_nodes $HOST_NUM -gpus $HOST_GPU_NUM \
2>&1 >>${MODEL_DIR}/logs/${TJ_INSTANCE_ID}.log

# Debug
#sleep 10000
echo "Finished training"
exit
