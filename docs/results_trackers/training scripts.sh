# prefix=sisfall_20hz_2s

# prefix=upfall_20hz_2s

# prefix=umafall_20hz_2s
# prefix=umafall_20hz_3s
# prefix=umafall_20hz_4s
# prefix=umafall_native_2s

# prefix=fallalld_20hz_2s
# prefix=fallalld_20hz_3s
# prefix=fallalld_20hz_5s
# prefix=fallalld_native_2s

# prefix=fallalld_native_2s


# prefix=upfall_native_2s

prefix=sisfall_20hz_3s

for fold in 0 1 2 3 4; do
  dataset="${prefix}_fold${fold}"

  python -m src.training.train_lstm_classifier --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_lstm_ae --dataset "$dataset" --model-seed 42 --device auto
done

# Shallow models

# Deep models 
  python -m src.training.train_cnn1d --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_lstm_classifier --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_dense_ae --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_lstm_ae --dataset "$dataset" --model-seed 42 --device auto


# Aggregating results
prefix=sisfall_20hz_5s
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model lstm_classifier --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model lstm_ae --model-seed 42

python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model random_forest --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model xgboost --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model isolation_forest --model-seed 42

python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model cnn1d --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model lstm_classifier --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model dense_ae --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model lstm_ae --model-seed 42

python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model cnn1d_ae --model-seed 42