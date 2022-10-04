GPUID=$1
OUTDIR=outputs/RAF-DB/
REPEAT=3
mkdir -p $OUTDIR

python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 10 100         --category gender --train_aug

python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 1 10 100         --category race --train_aug


