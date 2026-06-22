import json
import os
import subprocess
import shutil

def main():
    pub_india_dir = "/Users/kanhaiya_mehta/Downloads/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge"
    json_path = os.path.join(pub_india_dir, "sample_candidates.json")
    jsonl_path = os.path.join(pub_india_dir, "sample_candidates.jsonl")
    artifacts_dir = os.path.join(pub_india_dir, "sample_artifacts")
    output_csv = os.path.join(pub_india_dir, "sample_submission_output.csv")
    main_model_path = "/Users/kanhaiya_mehta/redrob-data/artifacts/ranker_model.pkl"
    
    print(f"1. Converting {json_path} -> {jsonl_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"2. Building features & embeddings index in {artifacts_dir}...")
    os.makedirs(artifacts_dir, exist_ok=True)
    subprocess.run([
        "./.venv/bin/python", "build_index.py",
        "--candidates", jsonl_path,
        "--artifacts", artifacts_dir,
        "--backend", "fastembed"
    ], check=True)
    
    print(f"3. Copying LightGBM model to {artifacts_dir}...")
    shutil.copy(main_model_path, os.path.join(artifacts_dir, "ranker_model.pkl"))
    
    print(f"4. Running rank.py on sample candidates -> {output_csv}...")
    subprocess.run([
        "./.venv/bin/python", "rank.py",
        "--candidates", jsonl_path,
        "--artifacts", artifacts_dir,
        "--out", output_csv,
        "--topk", str(len(data))
    ], check=True)
    
    print("Verification: reading first 5 rows of output:")
    import pandas as pd
    df = pd.read_csv(output_csv)
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
