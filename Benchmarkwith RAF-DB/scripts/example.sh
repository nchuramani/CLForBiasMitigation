GPUID=$1
OUTDIR=outputs/permuted_MNIST_incremental_domain
REPEAT=2
mkdir -p $OUTDIR

python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type mycnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 10 100         --category gender --train_aug

python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type mycnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 1 10 100         --category race --train_aug

python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type mycnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 1 10 100         --category age_combined --train_aug

