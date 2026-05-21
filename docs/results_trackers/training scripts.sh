# --------------------------------------
# Training scripts for all models and datasets
# --------------------------------------

# Traditional ML models
for model in  random_forest xgboost isolation_forest; do
  for prefix in sisfall_native_2s fallalld_native_2s umafall_native_2s upfall_native_2s; do
    for fold in 0 1 2 3 4; do
      dataset="${prefix}_fold${fold}"
      
      python -m src.training.train_"$model" --dataset "$dataset" --model-seed 42 --device auto
    done
  done
done

# Deep models
for model in cnn1d cnn1d_ae_large lstm_classifier lstm_ae; do
  for prefix in fallalld_native_2s umafall_native_2s upfall_native_2s; do
    for fold in 0 1 2 3 4; do
      dataset="${prefix}_fold${fold}"
      
      python -m src.training.train_"$model" --dataset "$dataset" --model-seed 42 --device auto
    done
  done
done

for model in cnn1d cnn1d_ae_large lstm_classifier; do
      
  python -m src.training.train_"$model" --dataset sisfall_native_2s_fold3 --model-seed 42 --device auto
done
  
  python -m src.training.train_cnn1d_ae_large --dataset "$dataset" --model-seed 42 --device auto

    python -m src.training.train_lstm_ae --dataset "$dataset" --model-seed 42 --device auto
    python -m src.training.train_lstm_classifier --dataset "$dataset" --model-seed 42 --device auto

    python -m src.training.train_cnn1d_large --dataset "$dataset" --model-seed 42 --device auto

# --------------------------------------
# aggregate metrics for the best models on the best dataset
# --------------------------------------

# Classification models
for dataset in fallalld umafall upfall; do
  for model in random_forest xgboost lstm_classifier cnn1d_large; do
  
  dataset_prefix="${dataset}_native_2s"
  
  python -m src.training.aggregate_cv_metrics \
  --dataset-prefix "$dataset_prefix" \
  --n-folds 5 \
  --mode classification \
  --model "$model" \
  --model-seed 42
  done  
done

# Anomaly detection models
for dataset in fallalld umafall upfall; do
  for model in isolation_forest cnn1d_ae_large lstm_ae; do
  
  dataset_prefix="${dataset}_native_2s"
  
  python -m src.training.aggregate_cv_metrics \
  --dataset-prefix "$dataset_prefix" \
  --n-folds 5 \
  --mode tsad \
  --model "$model" \
  --model-seed 42
  done  
done


python -m src.training.aggregate_cv_metrics \
  --dataset-prefix sisfall_native_2s \
  --n-folds 5 \
  --mode tsad \
  --model cnn1d_ae_large \
  --model-seed 42

python -m src.training.aggregate_cv_metrics \
  --dataset-prefix sisfall_native_2s \
  --n-folds 5 \
  --mode tsad \
  --model lstm_ae \
  --model-seed 42

python -m src.training.aggregate_cv_metrics \
  --dataset-prefix sisfall_native_2s \
  --n-folds 5 \
  --mode classification \
  --model lstm_classifier \
  --model-seed 42

  python -m src.training.aggregate_cv_metrics \
  --dataset-prefix sisfall_native_2s \
  --n-folds 5 \
  --mode classification \
  --model cnn1d \
  --model-seed 42