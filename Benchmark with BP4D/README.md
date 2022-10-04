## Preparation
```bash
pip install -r requirements.txt
```
Request access and download the BP4D dataset and put it under  ```BP4D/``` as the same level as main.py.
Adapt the data paths under ```dataloader/base.py``` accordingly.

## Example Script
The scripts for reproducing the results of this paper are under the scripts folder.
- 
- Example: Run all experiments with the Domain-Incremental scenario with BP4D.  
```bash
./scripts/example.sh 0
# The last number is gpuid
```
## Sampled python Script
Sample python3 command:
```bash
python3 -u main.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --no_class_remap --force_out_dim 7 --schedule 25 --batch_size 24 --model_type custom_cnn --model_name Net  --agent_type customization  --agent_name GEM_100        --lr 0.0001 --reg_coef 10 100  --category "gender" --train_aug
```
Update the scripts to choose:
1. ```--category``` from ```['gender', 'race']```.
2. ```--agent_name``` from ```[EWC, EWC_online, SI, MAS, Naive_Rehearsal, GEM]``` (see ```agents/customisation.py``` for actual examples). 
3. ```--model_type``` from ```[resnet, senet, lenet, mlp, custom_cnn]``` (see ```models/```).
4. `````--train_aug````` for running experiments with Data-Augmentation. 

Please refer to [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark) for more details.