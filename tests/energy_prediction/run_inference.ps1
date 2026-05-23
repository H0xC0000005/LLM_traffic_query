# run_inference.ps1
# Example usage for inference on ASHRAE GEPIII test set (Kaggle submission format)

$ErrorActionPreference = "Stop"

# Paths (edit as needed)
$DATA_PARQUET = "dataset\ashrae-energy-prediction\processed_features\ashrae_test_merged_plus_manual_features"                 # directory or file
$MODEL_DIR    = "E:\repos\LLM_traffic_query\tests\energy_prediction\lgbm_debug_no_ids\models\lgbm_20260306_164600_lr0.05_L127_minleaf300_exp_bid__exp1_b_s42_cache"         # contains lgbm_model.txt (+ optional manifests)
$OUT_CSV      = "submissions/exp1_b_s42.csv"

# Optional: activate venv
# .\.venv\Scripts\Activate.ps1

# switch with --use_expert_features
python .\inference.py `
  --data_parquet $DATA_PARQUET `
  --model_dir $MODEL_DIR `
  --out_csv $OUT_CSV `
  --invert_site0_meter0_to_kbtu `
  --use_expert_features `
  --clip_negative