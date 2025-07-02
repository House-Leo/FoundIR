## Two stage training script for FoundIR

## First stage: train the model on single degradation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=7689 train.py --meta ./MillionIRData_single_train_meta_info.txt

## Second stage: train the model on all degradations. Please uncomment Line49 and Line95 in train.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=7689 train.py --meta ./MillionIRData_train_meta_info.txt