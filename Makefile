PYTHON := python
REQUIREMENTS := requirements.txt
PATH_TO_MODEL := models/model.cbm
PATH_TO_TEST_MODEL_DATA := data/processed/test.csv
MODEL_PARAMS := models/model_params.json
TRAIN_DATA := data/features/prepared_train.csv


install:
	$(PYTHON) -m pip install -r $(REQUIREMENTS)


pull-data:
	echo "Pulling data starting!"
	dvc pull
	echo "Data pulled!"


transform-data: pull-data
	echo "Data transforming started!"
	$(PYTHON) src/features/transform.py --model_test $(PATH_TO_TEST_MODEL_DATA) --raw_data data/raw/ --not_save_model_test
	echo "Data transformed!"

select-params: $(TRAIN_DATA)
	echo "Hyperparameters selection started!"
	$(PYTHON) src/models/hyperparams_selection.py -d $(TRAIN_DATA) -o $(MODEL_PARAMS) -m models/model_metrics.json
	echo "Best hyperparameters selected!"


train-model: $(TRAIN_DATA) $(MODEL_PARAMS)
	echo "Train model on all data started!"
	$(PYTHON) src/models/train_model.py -d $(TRAIN_DATA) -p $(MODEL_PARAMS) -o models/model.cbm
	echo "Model fitted!"


test-model: $(PATH_TO_MODEL) $(PATH_TO_TEST_MODEL_DATA)
	$(PYTHON) src/models/model_predict.py -m $(PATH_TO_MODEL) -d $(PATH_TO_TEST_MODEL_DATA) -o data/processed/predictions.csv
	$(PYTHON) src/tests/test_model.py -p data/processed/predictions.csv -t data/processed/test.csv


all: install pull-data select-params train-model test-model

