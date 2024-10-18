# ME5418_ML

## Steps to Set Up

### Step1: Environment configuration

```
mkdir your_workspace
cd your_workspace/
git clone https://github.com/YukiKuma111/ME5418_ML.git
cd ME5418_ML/
conda env create -f environment.yaml

# test env
python env/group24_env.py
```

You can change movement mode by modifing `state` defined in [`env/group24_env.py`](./env/group24_env.py) line 1029.

If you don't want any obstacle, change `hardcore: bool = True` to __False__ in [`env/group24_env.py`](./env/group24_env.py) line 149