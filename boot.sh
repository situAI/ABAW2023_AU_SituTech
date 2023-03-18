
PRO_NAME=au_train
DEVICE=1
PORT=9001
NODE=1
SEED=3600
SEQLEN=128
EPOCHS=50

# for train
CONFIG=config/train.yaml
# for valid
# CONFIG=config/valid.yaml
# for infer
# CONFIG=config/infer.yaml

# set log
LOG_DIR=./logs
if [ ! -d "$LOG_DIR" ] ;
then
  echo "mkdir $LOG_DIR"
  mkdir -p "$LOG_DIR"
fi

function _echo_help_message(){
cat <<END
用法: $(bash boot.sh "$0")
  train   模型训练
  valid  模型验证
  infer  模型推理
END
}

function _train_something(){
    CUDA_VISIBLE_DEVICES=${DEVICE} nohup python -m torch.distributed.launch \
        --master_port ${PORT} --nproc_per_node=$NODE \
        main.py ${CONFIG} FeatSolver \
        --seed ${SEED} \
        --seq_len ${SEQLEN} \
        --epochs ${EPOCHS} \
        --name ${PRO_NAME} > ${LOG_DIR}/${PRO_NAME}.log 2>&1 &

}

function _valid_something(){
    CUDA_VISIBLE_DEVICES=${DEVICE} nohup python -m torch.distributed.launch \
        --master_port ${PORT} --nproc_per_node=$NODE \
        main.py ${CONFIG} ValidSolver \
        --seed ${SEED} \
        --seq_len ${SEQLEN} \
        --epochs ${EPOCHS} \
        --name ${PRO_NAME} > ${LOG_DIR}/${PRO_NAME}.log 2>&1 &

}

function _infer_something(){
    CUDA_VISIBLE_DEVICES=${DEVICE} nohup python -m torch.distributed.launch \
        --master_port ${PORT} --nproc_per_node=$NODE \
        main.py ${CONFIG} InferSolver \
        --seed ${SEED} \
        --seq_len ${SEQLEN} \
        --epochs ${EPOCHS} \
        --name ${PRO_NAME} > ${LOG_DIR}/${PRO_NAME}.log 2>&1 &

}


case "${1:-help}" in
    train) _train_something ;;
    valid) _valid_something ;;
    infer) _infer_something ;;
    # 其他
    *)  _echo_help_message ;;
esac
