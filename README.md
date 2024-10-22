# translate_eitb_parcc_to_english

This repository contains a Python script that translates the [EITB PARCC by Vicomtech eu-es parallel corpus](https://huggingface.co/datasets/Helsinki-NLP/eitb_parcc) from Spanish to English.

You can find the (partially) translated dataset [here](https://huggingface.co/datasets/itzune/eitb_parcc_en).


## Requirements

The following script has been tested with Ubuntu 24.04.1 LTS and Python 3.12.3.

You can meet the requirements by running the following command:

```bash
apt update && apt install python3-pip python3-venv
```

## Installation

Create a virtual environment and install the required packages by running the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

To be able to download certain datasets from Hugging Face, you need to have an account and an API key. You can create an account [here](https://huggingface.co/join) and get your API key [here](https://huggingface.co/settings).

Once you have your API key, you can set it as an environment variable by running the following command:

```bash
apt install git-lfs
git config --global credential.helper store
huggingface-cli login
```

### Set the dataset name

You can set the dataset name by changing the `dataset_name` variable in the `translate_eitb_parcc_to_en.py` script.

```python
dataset_name = "itzune/eitb_parcc_en"
```

## Usage

```bash
python3 translate_eitb_parcc_to_en.py
```

## License

This repository is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for more information.