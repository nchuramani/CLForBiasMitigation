## Preparation
```bash
pip install -r requirements.txt
```
Then you should download RAF-DB dataset and put it to the same directory with iBatchLearn.py   
## Demo
The scripts for reproducing the results of this paper are under the scripts folder.

- Example: Run all experiments in the incremental domain scenario with RAF-DB.  
```bash
./scripts/example.sh 0
# The last number is gpuid
```
## Bash Script
Simple python3 experiment example
```bash
python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type mycnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 10 100         --category gender --train_aug
```
you can select --category (gender, race, age_combined), --agent_name, --model_type, --train_aug
