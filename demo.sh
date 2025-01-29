#!/bin/bash
#Train x2
python main.py --dir_data ../../ --n_GPUs 4 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model NLSN --scale 2 --patch_size 96 --save NLSN_x2_tetwork --data_train DIV2K
python main.py --dir_data ../../ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model NLSN --scale 2 --patch_size 96 --save NLSN_x2_tetwork --data_train DIV2K

#NSNL with 64 features 50 epoches scale 2 Set 5 fix seed set5
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 2 --patch_size 96 --save NLSN_x2_testfix --data_train DIV2K

#NSNL with 64 features 50 epoches scale 4 Set 5 fix seed set5
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 4 --patch_size 96 --save NLSN_x4_testfix --data_train DIV2K

#NSNL with 64 features 50 epoches scale 2 Set 14 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 2 --patch_size 96 --save NLSN_x2_testfix_set14 --data_train DIV2K

#NSNL with 64 features 50 epoches scale 4 Set 14 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 4 --patch_size 96 --save NLSN_x4_testfix_set14 --data_train DIV2K

#NSNL with 64 features 50 epoches scale 2 B100 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 2 --patch_size 96 --save NLSN_x2_testfix_B100 --data_train DIV2K


#NSNL with 64 features 50 epoches scale 4 B100 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 4 --patch_size 96 --save NLSN_x4_testfix_B100 --data_train DIV2K

#NSNL with 64 features 50 epoches scale 2 Urban100 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model nlsn --scale 2 --patch_size 96 --save NLSN_x2_testfix_Urban100 --data_train DIV2K


#Proposed Method with 64 features 50 epoches scale 2 Set 5 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 2 --patch_size 96 --save final_model2_x2_testfix --data_train DIV2K

#Proposed Method with 64 features 50 epoches scale 4 Set 5 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_testfix --data_train DIV2K

#Proposed Method with 64 features 50 epoches scale 2 Set 14 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 2 --patch_size 96 --save final_model2_x2_testfix_set14 --data_train DIV2K


#Proposed Method with 64 features 50 epoches scale 4 Set 14 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_testfix_set14 --data_train DIV2K

#Proposed Method with 64 features 50 epoches scale 2 B100 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 2 --patch_size 96 --save final_model2_x2_testfix_B100 --data_train DIV2K

#Proposed Method with 64 features 50 epoches scale 4 B100 fix seed
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 50 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_testfix_B100 --data_train DIV2K

#Proposed Method with 64 features 100 epoches scale 4 Set5 
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 100 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_100ep --data_train DIV2K

#Proposed Method with 64 features 200 epoches scale 4 Set5 
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 200 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_200ep --data_train DIV2K

#Proposed Method with 64 features 300 epoches scale 4 Set5 
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 300 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_300ep --data_train DIV2K
#Proposed Method with 64 features 300 epoches scale 4 Set5
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 16 --model final_model2 --scale 4 --patch_size 96 --save final_model2_x4_300ep --data_train DIV2K

#Work My setting

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model final_model2 --scale 2 --patch_size 96 --save Final_Model2_x2_final_result --data_train DIV2K 

#Work My setting Scale 4

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model final_model2 --scale 2 --patch_size 96 --save Final_Model2_x4_final_result --data_train DIV2K 

#Work My setting Scale Jagrati_Model x4

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model jagrati_model --scale 4 --patch_size 96 --save Jagrati_Model_x4_Result --data_train DIV2K

#Work My setting Scale Jagrati_Model x3

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model jagrati_model --scale 3 --patch_size 96 --save Jagrati_Model_x3_Result --data_train DIV2K

#Work My setting Scale Jagrati_Model x2

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model jagrati_model --scale 2 --patch_size 96 --save Jagrati_Model_x2_Result --data_train DIV2K

#Work My setting #load pretrain

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model final_model2 --scale 2 --patch_size 96 --save Final_Model2_x2_final_result_continue-pre_train --data_train DIV2K --pre_train /home/vtrg/Desktop/Jagrati/Non-Local-Sparse-Attention_17_10_22/experiment/Final_Model2_x2_final_result_two/model/model_latest.pt


#Work My setting #load pretrain 2nd time

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model final_model2 --scale 2 --patch_size 96 --save Final_Model2_x2_final_res_continue-pre_train --data_train DIV2K --pre_train /home/vtrg/Desktop/Jagrati/Non-Local-Sparse-Attention_17_10_22/experiment/Final_Model2_x2_final_res/model/model_latest.pt


#Train x2 Baseline Setting
python main.py --dir_data ../../ --n_GPUs 4 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model NLSN --scale 2 --patch_size 96 --save NLSN_x2 --data_train DIV2K


#Not Work
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model FINAL_MODEL2 --scale 2 --patch_size 96 --save FINAL_MODEL2RESULT_x2 --data_train DIV2K

#Not-work copy

python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model FINAL_MODEL2 --scale 2 --patch_size 96 --save FINAL_MODEL2RESULT_x2 --data_train DIV2K

#Not Work-train3 300ep
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 300 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 8 --model test3 --scale 2 --patch_size 96 --save Test3_x2 --data_train DIV2K

#Not Work-train3 1000ep
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 8 --model test3 --scale 2 --patch_size 96 --save Test3_x2 --data_train DIV2K

#Work-train3 300ep
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 300 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 8 --model test3 --scale 2 --patch_size 96 --save Test3_x2 --data_train DIV2K

#Work-train3 1000ep
python main.py --dir_data ../../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 8 --model test3 --scale 2 --patch_size 96 --save Test3_x2 --data_train DIV2K


#Test2 Method with 64 features 300 epoches scale 2 Set5
python main.py --dir_data ../../ --n_GPUs 4 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 300 --chop --save_results --n_resblocks 32 --n_feats 64 --res_scale 0.1 --batch_size 8 --model test2 --scale 2 --patch_size 96 --save Test2_300ep_x2 --data_train DIV2K

#Train Scale 8
python main.py --dir_data ./../Dataset/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model nlsn --scale 8 --patch_size 96 --save NLSN_X8 --data_train DIV2K

#ChatGPT Train code scale 8

python main.py --dir_data /path/to/DIV2K/ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 8 --model nlsn --scale 8 --patch_size 96 --save NLSN_X8 --data_train DIV2K



#Test x2
python main.py --dir_data ../../ --model NLSN  --chunk_size 144 --data_test Set5+Set14+B100+Urban100 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train model_x2.pt --test_only 

python main.py --dir_data ../../Dataset/ --model NLSN  --chunk_size 144 --data_test Set5 --n_hashes 4 --chop --save_results --rgb_range 1  --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train ../../models/model_x2.pt --test_only 


#python main.py --dir_data ../../Dataset/ --model NLSN  --chunk_size 144 --data_test Set5+Set14+B100+Urban100 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train ../../models/model_x2.pt --test_only


##python main.py --dir_data ../../Dataset/ --model final_model2  --chunk_size 144 --data_test Set5+Set14+B100+Urban100 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 4 --n_feats 256 --n_resblocks 32 --res_scale 0.1 --batch_size 4 --pre_train ../../models/model_best.pt --test_only


#Testx3
python main.py --dir_data ../../Dataset/ --model jagrati_model  --chunk_size 144 --data_test Set5+Set14+B100+Urban100+Manga109 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 3 --n_feats 256 --n_resblocks 32 --res_scale 0.1 --batch_size 4 --pre_train ../../models/model_best.pt --test_only

#Testx2
python main.py --dir_data ../../Dataset/ --model jagrati_model  --chunk_size 144 --data_test Set5+Set14+B100+Urban100+Manga109 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1 --batch_size 4 --pre_train ../../models/model_best.pt --test_only


#Test
python main.py --dir_data ../../ --model NLSN  --chunk_size 144 --data_test Set5+Set14+B100+Urban100+Manga109 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train model_x2.pt --test_only 
