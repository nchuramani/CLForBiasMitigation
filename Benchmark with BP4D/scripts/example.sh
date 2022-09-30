GPUID=$1
OUTDIR=outputs/BP4D_GENDER_MAS
REPEAT=3
mkdir -p $OUTDIR

python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam  --category gender  --no_class_remap --force_out_dim 12 --schedule 5 --batch_size 24 --model_type mycnn --model_name Net  --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 100        | tee ${OUTDIR}/MAS.log



