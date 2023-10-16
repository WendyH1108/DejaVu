file=../wsc.jsonl
output_file=../test_output.jsonl


echo "start running ${file}"
export SPRARSE_PATH="../predictors/13b-sparse-predictor"
export LAYER=40
export TOPK=5000
export ATTN_TOPK_1=24
export ATTN_TOPK_2=48
export SPARSE_ATT=1

LAYER=40
TOPK=5000
ATTN_TOPK_1=24
ATTN_TOPK_2=48

ARGS="--model-name ../models/facebook_opt_13b \
--model-type opt-ml-att-sparse \
--seed 42 \
--fp16 \
--num-layers 40 \
--max-layers 96 \
--budget 22800 \
--num-iters 1000000 \
--dist-url tcp://127.0.0.1:9031 \
--token-micro-batch-size 1 \
--world-size 1 --pipeline-group-size 1 --data-group-size 1 \
--pp-mode pipe_sync_sample_mask_token_pipe \
--infer-data ${file} \
--output-path ${output_file}"

(trap 'kill 0' SIGINT; \
python dist_inference_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 0 \
    & \
wait)