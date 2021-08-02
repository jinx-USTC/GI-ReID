# GI-ReID
Code release for "Cloth-Changing Person Re-identification from A Single Image with Gait Prediction and Regularization". More files are coming soon.


Sorry for delay. Due to an unexpected "Reject" received from reviewer, here we only provide the metarials for Baseline now (including dataloader files for multiple colth-changing reid datasets). The rest parts about gait prediction and regularization will be released upon acceptance.

# Training and eval setting of Baseline w.r.t the different cloth-changing datasets

## LTCC standard eval setting (all training setting):
nohup python train_imgreid_xent_htri.py --gpu-devices 1 --root /data1/Datasets_jinx/Cloth-ReID/reid_datasets/ -s ltcc -t ltcc -a resnet50_fc512 --height 256 --width 128 --max-epoch 60 --stepsize 20 40 --lr 0.0003 --train-batch-size 32 --test-batch-size 100 --save-dir ../debug_dir/test_ltcc3 --train-with-all-cloth --use-standard-metric --eval-freq 10 > ./debug3.log 2>&1 &

## LTCC cloth-changing eval setting (all training setting):
### Note that, if use --use-cloth-changing-metric, must add (g_pids[order] == q_pid) & ((g_cloids[order] == q_cloid) | (g_camids[order] == q_camid)):
nohup python train_imgreid_xent_htri.py --gpu-devices 1 --root /data1/Datasets_jinx/Cloth-ReID/reid_datasets/ -s ltcc -t ltcc -a resnet50_fc512 --height 256 --width 128 --max-epoch 60 --stepsize 20 40 --lr 0.0003 --train-batch-size 32 --test-batch-size 100 --save-dir ../debug_dir/test_ltcc3 --train-with-all-cloth --use-cloth-changing-metric --eval-freq 10 > ./debug5.log 2>&1 &

## PRCC same-cloth eval setting:
### must using cuhk03 eval protocal, and did not remove anything:
nohup python train_imgreid_xent_htri.py --gpu-devices 1 --root /data1/Datasets_jinx/Cloth-ReID/reid_datasets/ -s prcc -t prcc -a resnet50_fc512 --height 256 --width 128 --max-epoch 60 --stepsize 20 40 --lr 0.0003 --train-batch-size 32 --test-batch-size 100 --save-dir ../debug_dir/test_prcc --same-clothes  --just-for-prcc-test --use-metric-cuhk03 --eval-freq 10 > ./debug8_prcc.log 2>&1 &

## PRCC cloth-changing eval setting:
nohup python train_imgreid_xent_htri.py --gpu-devices 1 --root /data1/Datasets_jinx/Cloth-ReID/reid_datasets/ -s prcc -t prcc -a resnet50_fc512 --height 256 --width 128 --max-epoch 60 --stepsize 20 40 --lr 0.0003 --train-batch-size 32 --test-batch-size 100 --save-dir ../debug_dir/test_prcc --cross-clothes  --just-for-prcc-test --use-metric-cuhk03 --eval-freq 10 > ./debug9_prcc.log 2>&1 &
