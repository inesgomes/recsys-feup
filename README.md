To run step by step:

`python data_cleaning_city.py --city Philadelphia`

`python predictions.py --city Philadelphia --factor 0.25 --k 5 --model svd`

`python explainability.py --city Philadelphia --k 10 --model svd`

To run full pipeline (default model is SVD):

`python src --city Philadelphia`