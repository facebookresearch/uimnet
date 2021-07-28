
.PHONY: setup install rsync show_env conda_env

SHELL:=/bin/bash

##,----------------------------
##| Experiments Shell Variables
##`----------------------------
EXPERIMENTS_ROOT?:=
DATASETS_ROOT?:=
UI_ROOT?:=
UI_REMOTE?:=
USER:=$$(whoami)
HOST:=$$(hostname)
UI_JUPYTER_SERVER_PORT?:=
UI_VISDOM_SERVER_PORT?:=
UI_HTTP_SERVER_PORT?:=

CURRENT_HOST?=

UI_LOCAL_DIR:=$(UI_ROOT)
UI_REMOTE_DIR:=$(UI_REMOTE):/private/home/$(USER)/work_dir
RSYNC_PARAMS:=-avz --progress --exclude '__pycache__' --exclude '*.pyc' --exclude '*.so' --exclude '*.o' --exclude '*.ipynb' --exclude '*.pdf' --exclude '*.pt' --exclude '*.h5' --exclude '*.hdf5' --filter=':- .gitignore' --exclude='/.git'

help:
	@echo "dev_install     - install the package to the active Python's site-packages"
	@echo "clean           - remove all build, test, coverage and Python artifacts"
	@echo "clean-build     - remove build artifacts"
	@echo "clean-pyc       - remove Python file artifacts"
	@echo "clean-test      - remove test and coverage artifacts"
	@echo "conda_env       - construct environment from environment.yaml"
	@echo "rsync           - rsyncs project root to project remote"
	@echo "fetch_notebooks - rsync notebooks from remote to host"
	@echo "push_notebooks  - rsync notebooks from host to remote"
	@echo "fetch_data      - rsync data from remote to host"
	@echo "push_data       - rsync data from host to remote"
	@echo "test            - Runs unit tests"
	# @echo "lint - check style with flake8"
	# @echo "profile - runs lineprofiler on profiling scripts"
	# @echo "test - run tests"
	# @echo "docs - generate Sphinx HTML documentation, including API docs"


show_env:
	@printf '\n%s\n'   "###################################"
	@printf '%s\n'	   "## Project environment Variables ##"
	@printf '%s\n' 	   "###################################"
	@printf '%s: %s\n' "Username" "$(USER)"
	@printf '%s: %s\n' "Project Root" "$(UI_ROOT)"
	@printf '%s: %s\n' "Experiments Root Path" "$(EXPERIMENTS_ROOT)"
	@printf '%s: %s\n' "Datasets Root" "$(DATASETS_ROOT)"
	@printf '%s: %s\n' "OMP threads" "$(OMP_NUM_THREADS)"
	@printf '%s: %s\n' "Project local" "$(HOST)"
	@printf '%s: %s\n' "Project remote" "$(UI_REMOTE)"
	@printf '%s: %s\n' "Project local dir" "$(UI_LOCAL_DIR)"
	@printf '%s: %s\n' "Project remote dir" "$(UI_REMOTE_DIR)"
	@printf '%s\n'     "Rsync params:"
	@printf '%s\n'     "$(RSYNC_PARAMS)"
	@printf '%s: %s\n' "jupyter server port" "$(UI_JUPYTER_SERVER_PORT)"
	@printf '%s: %s\n' "visdom server port" "$(UI_VISDOM_SERVER_PORT)"
	@printf '%s: %s\n' "http server port" "$(UI_HTTP_SERVER_PORT)"
	@printf '\n'


clean: clean-build clean-pyc clean-test clean-cpp

clean-build:
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -rf {} +
	@python setup.py clean --all
	@find exts/cython/src/ -name '*.c' -exec rm -rf {} +

clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-cpp:
	@find . -name '*.so' -exec rm -f {} +

clean-test:
	@rm -fr .tox/
	@rm -f .coverage
	@rm -fr htmlcov/

conda_env:
	@conda env create -f $(UI_ROOT)
	@conda env update

tests:
	@nose2 -C -vvv -s ./tests

build-ext: clean
	python3 setup.py build_ext --inplace

build: build-ext

dev_install: build
	@pip install -e .

rsync:
	@rsync $(RSYNC_PARAMS) $(UI_LOCAL_DIR) $(UI_REMOTE_DIR)

fetch_notebooks:
	@rsync -avz $(UI_REMOTE_DIR)/ui/notebooks/ $(UI_LOCAL_DIR)/notebooks

push_notebooks:
	@rsync -avz $(UI_LOCAL_DIR)/notebooks/ $(UI_REMOTE_DIR)/ui/notebooks

fetch_data:
	@rsync -avz $(UI_REMOTE_DIR)/ui/data/ $(UI_LOCAL_DIR)/data

push_data:
	@rsync -avz $(UI_LOCAL_DIR)/data/ $(UI_REMOTE_DIR)/ui/data

reset_checkpoints:
	@rm -rf checkpoints/ && mkdir -p ./checkpoints
