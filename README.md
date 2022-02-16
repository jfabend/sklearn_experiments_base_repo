# sklearn_experiments_base_repo
Base Repo for experimenting with sklearn and doing dataprep before

Installation after git clone:
python -m venv venv (...then select the venv as python interpreter.)
pip install -r .\requirements.txt (... to install all the necessary packages)
pip install -e . (... to load the code in this repo as package)
Add a file .env with ROOT_DIR="C:\yourpath\to\ja_pk" (... you find the folder ja_pk in src)

Data Prep:
Please change data path and path to data_pipeline.yml in ja_pk.dataprep.run_etl_pipeline.py.
Afterwards you can run this file and all configured preparation steps in data_pipeline.yml will be executed.

Experimenting:
Please change data path and path to experiment_config.yml in ja_pk.exp.experiment_controller.py.
Afterwards you can run this file to start your configured experiment.
