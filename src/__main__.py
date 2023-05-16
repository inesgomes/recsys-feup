import argparse
import os
import time
import pandas as pd
from surprise import SVD, NMF, BaselineOnly, KNNWithMeans

from data_cleaning_city import build_dataset
from predictions import train_test_split_surprise, train_model, explicit_evaluation, implicit_evaluation, trainset_df, get_recommendations
from explainability import explain_recs


algorithms = {
    'ALS': BaselineOnly(bsl_options={"method": "als", "n_epochs": 5, "reg_u": 12, "reg_i": 5}),
    'SVD5': SVD(n_factors=5),
    'SVD10': SVD(n_factors=10),
    'SVD15': SVD(n_factors=15),
    'SVD17': SVD(n_factors=17),
    'NMF10': NMF(n_factors=10),
    'KNN10': KNNWithMeans(k=10)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='SVD15')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--d', type=int, default=15)
    args = parser.parse_args()
    top_k = args.k
    
    # check if dataset exists first, if not create dataset
    filename = f"data/clean/reviews_{args.city}_2015_2020.pkl"
    df_review_clean = pd.read_pickle(filename) if os.path.exists(filename) else build_dataset(args.city)

    if df_review_clean.shape[0]==0:
        exit(-1)

    print(f"{args.city}")
    print(f"Total reviews: {df_review_clean.shape[0]:,}")
    print(f"Total businesses: {df_review_clean.business_id.nunique():,}")
    print(f"Total users: {df_review_clean.user_id.nunique():,}")
    
    # prepare train and test set
    trainset, testset, antitestset = train_test_split_surprise(df_review_clean, args.factor)
    print("\nTraining set: ")
    print(f"{(1-args.factor):.2%}")
    print(f"N ratings: {trainset.n_ratings:,}")
    print(f"N items: {trainset.n_items:,}")
    print(f"N users: {trainset.n_users:,}")
    print(f"Global mean: {trainset.global_mean:.2f}")

    # create predictions and recommendations 
    if args.model in algorithms:
        # for evaluation
        preds, recs = train_model(trainset, testset, algorithms[args.model], top_k)
        # for recommendations
        filtered_preds, _ = train_model(trainset, antitestset, algorithms[args.model], top_k)
    else:
        exit(-1)

    # evaluation
    print(f"\n{args.model} evaluation:")
    explicit_evaluation(preds)
    implicit_evaluation(recs, top_k)

    # explain recommendations
    print('\nExplain recommendations:')
    tic = time.time()
    train_df = trainset_df(trainset)
    unfiltered_matrix = pd.concat([
        pd.DataFrame(filtered_preds),
        train_df.rename(columns={'r': 'est'})
    ])
    print(f"d: {args.d}")
    ulfiltered_recs = get_recommendations(unfiltered_matrix, args.d)
    recs_interpretability, associations = explain_recs(ulfiltered_recs, train_df, top_k)

    toc = time.time()
    elapsed_time = toc-tic
    print(f" elapsed time {elapsed_time//3600:.0f}:{(elapsed_time%3600)//60:.0f}:{elapsed_time%60:.0f}")
    print(f"Number of associations created: {associations.shape[0]}")
    print(f"Model Fidelity = {recs_interpretability.dropna().shape[0]/recs_interpretability.shape[0]:.2%}")

    recs_interpretability.to_pickle(f'data/final/recs_interpretation_svd_{args.city}_top{top_k}.pkl')
   