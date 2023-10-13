for l in $(seq 0 1 40)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_att.py --dataset c4 --lr 0.0001 --k 0.3 --L ${l} & \
    wait)
done