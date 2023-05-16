import argparse
import pandas as pd
from apyori import apriori
from tqdm import tqdm

# check: 
# https://www.kaggle.com/code/mervetorkan/association-rules-with-python
# https://www.kaggle.com/code/pierrelouisdanieau/recommender-system-associations-rules

# constants
MIN_SUPP = 0.1
MIN_CONF = 0.1
MIN_LIFT = 0.1

def explain_recs(recs, trainset, top_k):
    """
    explaining recommendations based on the paper: 
    Explanation Mining: Post Hoc Interpretability of Latent Factor Models for Recommendation Systems
    """
    # step 2: create association rules
    results = pd.DataFrame(list(apriori(recs['recommendations'])))
    cols = ['items_base', 'items_add', 'confidence', 'lift']
    df_associations=results.explode('ordered_statistics')

    df_associations[cols] = pd.DataFrame(
        df_associations['ordered_statistics'].to_list(),
        columns = cols
    )
    df_associations.drop(columns='ordered_statistics', inplace=True)

    # filter according to rules: min_supp = 0.1, min_conf = 0.1, min_lift = 0.1, max_len= 2
    df_associations_filter = df_associations[
        (df_associations.apply(lambda x: len(x['items_base'])==len(x['items_add']), axis=1)) & 
        (df_associations['support']>=MIN_SUPP) &
        (df_associations['confidence']>=MIN_CONF) &
        (df_associations['lift']>=MIN_LIFT)
        ].copy(deep=True).drop_duplicates()

    df_associations_filter['X'] = df_associations_filter['items_base'].apply(lambda x: list(x)[0])
    df_associations_filter['Y'] = df_associations_filter['items_add'].apply(lambda x: list(x)[0])
    df_associations_filter.drop(columns=['items_base', 'items_add', 'items'], inplace=True)

    final_recs = recs.drop(columns=['actual']).explode('recommendations')
    rec_interpretations = pd.DataFrame()

    # step 3: for each user
    for uid, rec_u in tqdm(final_recs.groupby('user_id')):
        # step 4: compute the list {unseen} of items Y where X ⇒ Y if X ∈ {train} and Y not {train}
        # step 5: order unseen by supp/conf/lift
        iid_user = trainset[trainset['uid'] == uid]['iid'].drop_duplicates()
        # we do not recommend things that the user has seen
        rec_filtered = rec_u[~rec_u['recommendations'].isin(iid_user)]
        # the user is not new in the test set
        if iid_user.shape[0] > 0:
            assoc_user = df_associations_filter[(df_associations_filter['X'].isin(iid_user)) & 
                                                (~df_associations_filter['Y'].isin(iid_user))
                                                ].drop_duplicates()

            # step 6: merge with recommendations, order and select max N recs
            recs_assoc_u = pd.merge(rec_filtered, assoc_user, left_on='recommendations', right_on='Y', how='left')
            rec_assoc_u_top = recs_assoc_u.sort_values(by=['support', 'confidence', 'lift'], ascending=False)[:top_k]
            rec_interpretations = pd.concat([rec_interpretations, rec_assoc_u_top[['user_id', 'recommendations', 'X']]])

    rec_interpretations.columns = ['user_id', 'recommendation', 'explanation']
    return rec_interpretations, df_associations_filter


def main(city: str, model: str, k:int):
    """
    """
    trainset = pd.read_pickle("../data/clean/train.pkl")
    # step 1: recs variable
    recs = pd.read_pickle(f"../data/clean/recommendations_{model}_{city}_top_{k}.pkl")
    # interpret
    recs_interpretation, _ = explain_recs(recs, trainset)

    print(f"recs with post-hoc interpretation: {recs_interpretation.dropna().shape[0]/recs_interpretation.shape[0]:.2%}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()

    main(args.city, args.model, args.k)