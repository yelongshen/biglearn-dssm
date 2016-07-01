# biglearn-dssm

example training data:
	data/train_data_40k.tsv
	format:
	[query][\t][document]

example test data:
	data/test_data_clean.tsv
	[query][\t][document]

build:
	make

training example:
	binary/DSSM -train
	output model to : model/dssm.v2.model

prediction example:
	binary/DSSM -predict
	output result to data/test_data.v2.result
	
evaluation:
	cd data/
	python eval_auc.py test_data.v2.result test_data_clean.tsv
