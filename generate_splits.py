import json
import argparse

def generate_splits(output_path): 
    ignore = [452, 485, 490] 
    
    splits = {
        "train": range(1, 421),
        "val": range(421, 501), 
        "test": range(501, 661),
    }  
    
    splits["trainval"] = list(splits["train"]) + list(splits["val"])
    splits["all"] = list(splits["trainval"]) + list(splits["test"])

    for key in splits:
        splits[key] = [f"RibFrac_{x:03}" for x in splits[key] if x not in ignore]
    
    output = [{"train": splits["train"], "val": splits["val"]}] 
    
    with open(output_path, 'w') as f:
        json.dump(output, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Path to save the splits JSON file")
    args = parser.parse_args()
    
    generate_splits(args.output_path)
