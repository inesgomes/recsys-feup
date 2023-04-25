import argparse
import pandas as pd

# check: 
# https://www.kaggle.com/code/mervetorkan/association-rules-with-python
# https://www.kaggle.com/code/pierrelouisdanieau/recommender-system-associations-rules

def main(city: str, model: str):
    """
    assuming SVD Colaborative Filtering
    """
    print("TODO")
    preds = pd.read_pickle(f"../data/clean/predictions_{model}_{city}.pkl")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    main(args.city, args.model)