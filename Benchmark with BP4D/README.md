## Preparation
```bash
pip install -r requirements.txt
```
[Request access](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and download the BP4D dataset and put it under  ```BP4D/``` as the same level as main.py.
Adapt the data paths under ```dataloader/base.py``` accordingly.

## Example Script
The scripts for reproducing the results of this paper are under the scripts folder.
- 
- Example: Run EWC experiments with regularization coefficients 1 10 100 with the Domain-Incremental scenario with BP4D.  
```bash
./scripts/example.sh 0
# The last number is gpuid
```
## Sampled python Script
Sample python3 command:
```bash
python3 -u main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type customization  --agent_name EWC  --lr 0.0001 --reg_coef 10 100  --category gender --train_aug
```

Update the scripts to choose:
1. ```--category``` from ```['gender', 'race']```.
2. ```--agent_name``` from ```[EWC, EWC_online, SI, MAS, Naive_Rehearsal, GEM]``` (see ```agents/customisation.py``` for actual examples). 
3. ```--model_type``` from ```[resnet, senet, lenet, mlp, custom_cnn]``` (see ```models/```).
4. `````--train_aug````` for running experiments with Data-Augmentation. 

Please refer to [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark) for more details.

This will generate corresponding result text files under ```results/``` for model performance scores.
## Evaluation

Evaluation code for gender: 
```bash
python3 bp4d_gender_eval.py
```
Evaluation code for race:
```bash
python3 bp4d_race_eval.py
```

These read the result text files under ```results/``` and print BWT scores and the mean/std of the accuracies & fairness scores of the experiments.
