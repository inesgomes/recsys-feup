import pandas as pd
from surprise import SVD, Reader, Dataset, KNNWithMeans
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mse
import argparse


def split_train_test(df_review: pd.DataFrame, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    currently not being used
    Split the interactions based on the date
    """
    # if multiple reviews from the same user to a given business, select only the most recent
    df_review = df_review.sort_values(by="review_date").drop_duplicates(
        subset=['business_id', 'user_id'], keep='last')
    # split by date
    df_train = df_review[df_review['review_date']
                         < date].drop(columns='review_date')
    df_test = df_review[df_review['review_date']
                        >= date].drop(columns='review_date')
    # we select only users and businesses on test data that exist in the training data so that we don't suffer from the cold start problem
    df_test = df_test[(df_test['user_id'].isin(df_train['user_id'].unique())) &
                      (df_test['business_id'].isin(df_train['business_id'].unique()))]

    return df_train, df_test


def train_test_split_surprise(df_review: pd.DataFrame, factor: float):
    """
    """
    # maintain order by date
    df_review.sort_values(by='review_date', inplace=True)
    relevant_cols = ["user_id", "business_id", "review_stars"]
    # transform in surprise dataset
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df_review[relevant_cols], reader)
    # split x% to test
    trainset, testset = train_test_split(dataset, shuffle=False, test_size=factor)
    return trainset, testset


def get_user_top_n(matrix: pd.DataFrame, user_id: str, top: int):
    """
    function from: https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb
    """
    recommended_items = pd.DataFrame(matrix.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values(
        'predicted_rating', ascending=False)
    recommended_items = recommended_items.head(top)
    return recommended_items.index.tolist()


def get_recommendations(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    this is stupid, because we get always the top user recommendations?
    """
    # transform data in dataframe
    data = pd.DataFrame(data).drop("details", axis=1)
    data.columns = ['user_id', 'business_id', 'actual', 'predictions']
    
    # create matrix of predicted values
    data_matrix = data.pivot_table(
        index='user_id', columns='business_id', values='predictions').fillna(0)
    # mean user predictions
    data_lst = data.groupby('user_id', as_index=False)['business_id'].agg(
        {'actual': (lambda x: list(set(x)))}).set_index("user_id")
    # make recommendations for all members in the data
    recs = []
    for user in data_lst.index:
        predictions = get_user_top_n(data_matrix, user, n)
        recs.append(predictions)
    data_lst['recommendations'] = recs

    return data_lst


def train_svd(trainset: pd.DataFrame, testset: pd.DataFrame, n:int):
    """
    train and evaluate SVD (surprise lib)
    """
    # fit model
    algo = SVD(n_factors=10) # TODO change the number of factors
    algo.fit(trainset)

    # make predictions with lib
    test_pred = algo.test(testset)
    
    # get recommendations
    test_recs = get_recommendations(test_pred, n)

    return test_pred, test_recs


def train_knn(trainset: pd.DataFrame, testset: pd.DataFrame, n:int):
    """
    train and evaluate SVD (surprise lib)
    """
    # TODO hyperparameter tunning
    # fit model
    algo = KNNWithMeans(k=20)
    algo.fit(trainset)

    # make predictions with lib
    test_pred = algo.test(testset)
    
    # get recommendations
    test_recs = get_recommendations(test_pred, n)

    return test_pred, test_recs


def explicit_evaluation(preds):
    """
    """
    mse(preds)
    rmse(preds)


def apk(row):
    """
    Average Precision (AP): This measures the average precision, taking into account both the relevance 
    and the ranking of the recommended items.
    """
    y_pred = row.recommendations
    y_true = row.actual
    correct_predictions = 0
    running_sum = 0
    
    for i, yp_item in enumerate(y_pred):
        k = i+1 # our rank starts at 1
        if yp_item in y_true:
            correct_predictions += 1
            running_sum += correct_predictions/k
    return running_sum/len(y_true)


def implicit_evaluation(recs, n):
    """
    """
    # MAP@K
    mapk = recs.apply(apk, axis=1).mean()
    print(f"MAP@{n}: {mapk:.2%}")

    # Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    recs["n_relevant"] = recs.apply(lambda row: len(set(row.recommendations).intersection(row.actual)), axis=1)
    recs["n_actual"] = recs.actual.apply(len)
    pak = recs["n_relevant"].sum()/(recs.shape[0]*n)
    print(f"Precision@{n}: {pak:.2%}")

    # Recall@K: This measures the proportion of relevant items that are recommended in the top K items.
    rak = recs["n_relevant"].sum()/(recs['n_actual'].sum())
    print(f"Recall@{n}: {rak:.2%}")


def main(city: str, factor: float, k:int):
    """
    calculate predictions
    """
    # read clean data
    df_review = pd.read_pickle(f'../data/clean/reviews_{city}_2015_2020.pkl')

    # split
    trainset, testset = train_test_split_surprise(df_review, factor)

    # train and evaluate several models

    print("\n--- Popularity ---")
    # TODO: copy from recmetrics example notebook

    print("\n--- SVD (collaborative filtering)---")
    preds, recs = train_svd(trainset, testset, k)
    explicit_evaluation(preds)
    implicit_evaluation(recs, k)

    print("\n--- KNN with means (collaborative filtering)---")
    preds, recs = train_knn(trainset, testset, k)
    explicit_evaluation(preds)
    implicit_evaluation(recs, k)

    print("\n--- LDA (Content Base Filtering) ---")
    # TODO: search for the yelp recsys blogpost with topic modeling

    print("\n LightFM (Hybrid approach)")
    # TODO: check if this makes sense


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--factor', type=float, default=0.25)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    main(args.city, args.factor, args.k)
