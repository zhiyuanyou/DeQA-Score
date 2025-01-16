export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=./:$PYTHONPATH

python src/evaluate/iqa_eval.py \
	--level-names excellent good fair poor bad \
	--model-path checkpoints/DeQA-Score-LoRA-Mix3/ \
	--model-base ../ModelZoo/mplug-owl2-llama2-7b/ \
	--save-dir results/res_deqa_lora_mix3/ \
	--preprocessor-path ./preprocessor/ \
	--root-dir ../Data-DeQA-Score/ \
	--meta-paths ../Data-DeQA-Score/KONIQ/metas/test_koniq_2k.json \
				 ../Data-DeQA-Score/SPAQ/metas/test_spaq_2k.json \
				 ../Data-DeQA-Score/KADID10K/metas/test_kadid_2k.json \
				 ../Data-DeQA-Score/PIPAL/metas/test_pipal_5k.json \
				 ../Data-DeQA-Score/LIVE-WILD/metas/test_livew_1k.json \
				 ../Data-DeQA-Score/AGIQA3K/metas/test_agiqa_3k.json \
				 ../Data-DeQA-Score/TID2013/metas/test_tid2013_3k.json \
				 ../Data-DeQA-Score/CSIQ/metas/test_csiq_866.json \
