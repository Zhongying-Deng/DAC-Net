DATA=$1
CUDA_VISIBLE_DEVICES=0 python tools/train.py --root $DATA --trainer DACNet \
	--source-domains cartoon art_painting photo --target-domains sketch \
	--dataset-config-file configs/datasets/da/pacs_ca.yaml --config-file configs/trainers/da/dacnet/pacs.yaml \
	--output-dir output/dacnet_pacs/sketch --seed 2 \
	TRAINER.DACNET.WEIGHT_D .3 TRAINER.DACNET.LOSS_TYPE L1 TRAINER.DACNET.WEIGHT_CON .1 TRAIN.CHECKPOINT_FREQ 10
