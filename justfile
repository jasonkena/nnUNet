set export
# use scratch directory because NFS is painfully slow
nnUNet_raw := "/scratch/adhinart/nnUNet_raw"
nnUNet_preprocessed := "/scratch/adhinart/nnUNet_preprocessed"
nnUNet_results := "/scratch/adhinart/nnUNet_results"
nnUNet_n_proc_DA := "32"
nnUNet_keep_files_open := "True"

default:
    just --list

dataset:
    python /data/adhinart/ribseg/nnUNet/nnunetv2/dataset_conversion/Dataset011_ribseg.py

# RAM intensive
preprocess:
    nnUNetv2_plan_and_preprocess -d 11 --verify_dataset_integrity -np 32 16 32

generate_splits:
    python generate_splits.py /data/adhinart/ribseg/nnUNet/nnUNet_preprocessed/Dataset011_RibSeg/splits_final.json

train_2d:
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 11 2d 0 --npz --c

train_3d_fullres:
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 11 3d_fullres 0 --npz --c

train_3d_lowres:
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 11 3d_lowres 0 --npz --c

# this depends on 3d_lowres
train_3d_cascade_fullres:
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 11 3d_cascade_fullres 0 --npz --c

benchmark:
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 11 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs
    # CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 11 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading

inference_3d_fullres:
    # note that this only generates inference results on test
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i /scratch/adhinart/nnUNet_raw/Dataset011_RibSeg/imagesTs -o /scratch/adhinart/nnUNet_results/fullres_inference -d 11 -f 0 -c 3d_fullres
