from utils.parse_config import get_config
import os
from datetime import date

def save_metrics(epochs:list[int]=[],loss:list[float]=[],accuracy:list[float]=[],f1score:list[float]=[]):

    config = get_config()
    seed = config["seed"]
    file = config["result_file"]
    model = config["train"]
    config_model = config["model"][model]
    config_training = config["training"][model]

    id_date = str(date.today())
    title = f"### Trained model : {model} - Date : {id_date} \n"
    content = f"""| Epoch    | Accuracy | F1 score | Loss | Training Time (s) |
|-----------|-----------|-------|----------|---------------|
"""
    for (e,l,a,f) in zip(epochs,loss,accuracy,f1score):
        content +=  f"|{e}|{(100*a):>0.1f}%|{(100*f):>0.1f}%|{l}|-|\n"

    config_md = f"Config model : {str(config_model)} \nConfig training : {str(config_training)}\n\n"

    md_content = title + content + config_md
    with open(file,"a") as f:
        f.write(md_content)
        
        
