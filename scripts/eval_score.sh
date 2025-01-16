export PYTHONPATH=./:$PYTHONPATH

res_dir=./results/res_deqa_mix3/
gt_dir=../Data-DeQA-Score/

python src/evaluate/cal_plcc_srcc.py \
	--level_names excellent good fair poor bad \
	--pred_paths $res_dir/test_koniq_2k.json \
				 $res_dir/test_spaq_2k.json \
				 $res_dir/test_kadid_2k.json \
				 $res_dir/test_pipal_5k.json \
				 $res_dir/test_livew_1k.json \
				 $res_dir/test_agiqa_3k.json \
				 $res_dir/test_tid2013_3k.json \
				 $res_dir/test_csiq_866.json \
	--gt_paths  $gt_dir/KONIQ/metas/test_koniq_2k.json \
				$gt_dir/SPAQ/metas/test_spaq_2k.json \
				$gt_dir/KADID10K/metas/test_kadid_2k.json \
				$gt_dir/PIPAL/metas/test_pipal_5k.json \
				$gt_dir/LIVE-WILD/metas/test_livew_1k.json \
				$gt_dir/AGIQA3K/metas/test_agiqa_3k.json \
				$gt_dir/TID2013/metas/test_tid2013_3k.json \
				$gt_dir/CSIQ/metas/test_csiq_866.json \
