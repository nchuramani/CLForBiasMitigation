GPUID=$1
OUTDIR=outputs/BP4D_GENDER_MAS
REPEAT=3
mkdir -p $OUTDIR

python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam  --category gender  --no_class_remap --force_out_dim 12 --schedule 5 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type regularization --agent_name EWC   --lr 0.0001 --reg_coef 1 10 100 --category gender  

python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam  --category gender  --no_class_remap --force_out_dim 12 --schedule 5 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type regularization --agent_name EWC   --lr 0.0001 --reg_coef 1 10 100 --category gender --train_aug

python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam  --category gender  --no_class_remap --force_out_dim 12 --schedule 5 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type regularization --agent_name EWC   --lr 0.0001 --reg_coef 1 10 100 --category race

python3 -u ../main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam  --category gender  --no_class_remap --force_out_dim 12 --schedule 5 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type regularization --agent_name EWC   --lr 0.0001 --reg_coef 1 10 100 --category race --train_aug
