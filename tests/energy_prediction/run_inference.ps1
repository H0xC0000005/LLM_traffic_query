# run_inference.ps1
# Example usage for inference on ASHRAE GEPIII test set (Kaggle submission format)

$ErrorActionPreference = "Stop"

# Paths (edit as needed)
$DATA_PARQUET = "dataset\ashrae-energy-prediction\processed_features\ashrae_test_merged_plus_manual_features"                 # directory or file
$MODEL_DIR    = "lgbm_debug_no_ids\models\lgbm_20260305_130445_lr0.05_L127_minleaf300_exp__exp1_s44"         # contains lgbm_model.txt (+ optional manifests)
$OUT_CSV      = "submissions/submission_lgbm_run_001.csv"

# Optional: activate venv
# .\.venv\Scripts\Activate.ps1

python .\inference.py `
  --data_parquet $DATA_PARQUET `
  --model_dir $MODEL_DIR `
  --out_csv $OUT_CSV `
  --use_expert_features `
  --clip_negative