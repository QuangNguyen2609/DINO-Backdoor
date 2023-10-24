coco_path=$1
python main_harry.py \
	--output_dir logs/DINO_MTSD/4s_SWIN_BLUE_MULTI_24K -c config/DINO/DINO_4scale_swin.py --coco_path datasets/ \
	--resume logs/DINO_MTSD/4s_SWIN_BLUE_MULTI_24K/checkpoint0001.pth \
    --pretrain_model_path checkpoint0029_4scale_swin.pth \
    --finetune_ignore label_enc.weight class_embed \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
