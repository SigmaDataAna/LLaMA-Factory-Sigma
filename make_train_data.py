import json
import pandas as pd

if __name__ == "__main__":
    data = pd.read_parquet("train.parquet")
    print(data.iloc[0])

    results = []
    for idx, d in data.iterrows():
        tmp = {}
        tmp["instruction"] = "Please reason step by step, and put your final answer within \\boxed{}."
        tmp['input'] = d['problem']
        tmp['output'] = d['solution']
        results.append(tmp)
    
    with open("numina_math_cot.json", "w") as f:
        json.dump(results, f, indent=4)
