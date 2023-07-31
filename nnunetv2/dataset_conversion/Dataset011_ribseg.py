import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def copy_files(id, img_in_folder, gt_in_folder, img_out_folder, gt_out_folder):
    padded_id = str(id).zfill(3)
    shutil.copy(join(img_in_folder, f"RibFrac{id}-image.nii.gz"), join(img_out_folder, f"RibFrac_{padded_id}_0000.nii.gz"))
    shutil.copy(join(gt_in_folder, f"RibFrac{id}-rib-seg.nii.gz"), join(gt_out_folder, f"RibFrac_{padded_id}.nii.gz"))

if __name__ == "__main__":
    dataset_name = "Dataset011_RibSeg"
    
    input_folder = "/data/adhinart/ribseg/dataset/vol/"
    gt_folder = "/data/adhinart/ribseg/dataset/gt/"
    
    image_train_folder = join(nnUNet_raw, dataset_name, "imagesTr")
    image_test_folder = join(nnUNet_raw, dataset_name, "imagesTs")
    # image_val_folder = join(nnUNet_raw, dataset_name, "imagesVal")
    label_train_folder = join(nnUNet_raw, dataset_name, "labelsTr")
    label_test_folder = join(nnUNet_raw, dataset_name, "labelsTs")
    # label_val_folder = join(nnUNet_raw, dataset_name, "labelsVal")

    maybe_mkdir_p(image_train_folder)
    maybe_mkdir_p(image_test_folder)
    # maybe_mkdir_p(image_val_folder)
    maybe_mkdir_p(label_train_folder)
    maybe_mkdir_p(label_test_folder)
    # maybe_mkdir_p(label_val_folder)

    splits = {
        "train": range(1, 421),
        "val": range(421, 501),
        "test": range(501, 661),
    }

    # NOTE: training and validation go in same folder, splitting is done in generate_splits.py
    splits["trainval"] = list(splits["train"]) + list(splits["val"])

    with Pool(8) as p:
        res_tr = p.starmap_async(copy_files, [(id, input_folder, gt_folder, image_train_folder, label_train_folder) for id in splits["trainval"]])
        # res_val = p.starmap_async(copy_files, [(id, input_folder, gt_folder, image_val_folder, label_val_folder) for id in splits["val"]])
        res_ts = p.starmap_async(copy_files, [(id, input_folder, gt_folder, image_test_folder, label_test_folder) for id in splits["test"]])

        _ = res_tr.get() 
        # _ = res_val.get()  
        _ = res_ts.get()  

    labels = {str(i): i for i in range(1, 25)}
    labels["background"] = 0
    generate_dataset_json(join(nnUNet_raw, dataset_name), {0:"CT"}, labels, len(splits["trainval"]), '.nii.gz', dataset_name=dataset_name)
