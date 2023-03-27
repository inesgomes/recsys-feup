import pandas as pd
import argparse


def read_chunks(file, cols, chunk_size=500000):
    """
    read dataset in chunks
    """
    # load dataset
    df = pd.read_json(
        path_or_buf=f'../data/external/yelp_dataset/yelp_academic_dataset_{file}.json', chunksize=chunk_size, lines=True)
    # There are multiple chunks to be read
    chunk_list = [chunk[cols] for chunk in df]
    # return as dataframe
    return pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)


def data_cleaning(df_business: pd.DataFrame, df_review: pd.DataFrame, df_user: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    data cleaning and merging: selects only one city and merges the three datasets into one
    """
    # constants
    MIN_USER_REVIEWS = 5
    MIN_BUSI_REVIEWS = 2
    MIN_DATE = '2015-01-01'
    MAX_DATE = '2020-01-01'

    # column rename
    df_business.columns = 'business_'+df_business.columns
    df_review.columns = 'review_'+df_review.columns
    df_user.columns = 'user_'+df_user.columns
    # reviews between 2015 and 2020 - before covid and when we have more info
    df_review_date = df_review[(df_review['review_date'] >= MIN_DATE) & (
        df_review['review_date'] < MAX_DATE)]
    # select one city - because it's more realistic that people will want recommendations within their city
    df_business_city = df_business[df_business['business_city'] == city]
    # Inner merge with edited business file so only reviews related to the business remain
    df_review_clean = pd.merge(df_business_city, df_review_date,
                               left_on='business_business_id', right_on='review_business_id', how='inner')
    # merge with user info
    df_review_clean_enriched = pd.merge(
        df_review_clean, df_user, left_on='review_user_id', right_on='user_user_id', how='inner')
    # rename IDs
    df_review_clean_enriched.rename(columns={
                                    'business_business_id': 'business_id', 'user_user_id': 'user_id'}, inplace=True)
    # remove business with less than x ratings and users with less than y ratings - to remove noise and outliers
    df_review_clean_enriched = df_review_clean_enriched.groupby(
        "business_id").filter(lambda x: x['business_id'].count() >= MIN_BUSI_REVIEWS)
    df_review_clean_enriched = df_review_clean_enriched.groupby(
        "user_id").filter(lambda x: x['user_id'].count() >= MIN_USER_REVIEWS)
    # drop extra columns
    return df_review_clean_enriched.drop(columns=['business_city', 'review_business_id', 'review_user_id'])


def build_user_profile(df_review: pd.DataFrame):
    """
    count the number of friends, the number of years being elite and the number of days in yelp
    drop the raw columns
    """
    df_review['user_yelping_days'] = (
        df_review['review_date'] - pd.to_datetime(df_review['user_yelping_since'])).dt.days
    df_review['user_n_friends'] = df_review['user_friends'].apply(
        lambda x: len(x.split(",")))
    df_review['user_n_elite'] = df_review['user_elite'].apply(
        lambda x: len(x.split(",")))
    return df_review.drop(columns=['user_friends', 'user_elite', 'user_yelping_since'])


def build_business_profile(df_review: pd.DataFrame):
    """
    business content: topics OHE? maybe LDA or something - think about this
    """
    df_review['business_categories_lst'] = df_review['business_categories'].apply(
        lambda x: x.split(","))

    return df_review.drop(columns=['business_categories'])


def main(city: str):
    """
    Clean dataset and return reviews for one city only
    """
    # read data
    df_business = read_chunks('business', ['business_id', 'city', 'categories'])
    df_review = read_chunks('review', ['user_id', 'business_id', 'stars', 'date'])
    user_cols = ['user_id', 'review_count', 'yelping_since', 'useful',
                 'funny', 'cool', 'elite', 'friends', 'fans', 'average_stars']
    df_user = read_chunks('user', user_cols)

    # clean data
    df_review_clean = data_cleaning(df_business, df_review, df_user, city)

    # build user profile
    df_review_clean = build_user_profile(df_review_clean)

    # TODO: build business profile

    # save data
    df_review_clean.to_pickle(f"../data/clean/reviews_{city}_2015_2020.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True)
    args = parser.parse_args()
    main(args.city)
