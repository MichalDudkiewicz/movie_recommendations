import math
import multiprocessing
from multiprocessing import Process

import pandas as pd
import numpy as np
import statistics

# train and task csv columns
index_col = "index"  # int
user_col = "user_id"  # int
movie_col = "movie_id"  # int
rating_col = "rating"  # int/NA(float), 0-5 inclusive


def load_csv_to_df(csv_path):
    # ordering of user_id and movie_id can be changed here
    df = pd.read_csv(csv_path, delimiter=";", index_col=0, names=[index_col, user_col, movie_col, rating_col])
    return df


def sort_mean_sq_error(user_to_error):
    return user_to_error[1]


def get_similar_users(user_id, train_df, users, user_to_similar_users):
    print("get similar user for: " + str(user_id))
    user_movies = train_df.get_group(user_id)
    movie_ids = user_movies[movie_col].unique()
    similar_users = []
    for other_user_id in users:
        if user_id == other_user_id:
            continue

        other_user_movies = train_df.get_group(other_user_id)
        other_movie_ids = other_user_movies[movie_col].unique()

        common_movie_ids = np.intersect1d(movie_ids, other_movie_ids, assume_unique=True)

        mean_sq_err = sum(
            [(other_user_movies.loc[other_user_movies[movie_col] == movie_id][rating_col].iloc[0] -
              user_movies.loc[user_movies[movie_col] == movie_id][
                  rating_col].iloc[0]) ** 2
             for movie_id in common_movie_ids]) / len(common_movie_ids)

        similar_users.append((other_user_id, mean_sq_err))

    similar_users.sort(key=sort_mean_sq_error)
    user_to_similar_users[user_id] = [user_to_error[0] for user_to_error in similar_users]


def validate_n(task_validation_df, validation_df, user_to_similar_users, train_df, N, error, final_n):
    print("validate n: " + str(N))
    task_validation_df_copy = task_validation_df.copy()
    for ind in task_validation_df_copy.index:
        user_id = task_validation_df_copy[user_col][ind]
        movie_id = task_validation_df_copy[movie_col][ind]
        similar_users = user_to_similar_users[user_id]

        ratings = []
        n = 0
        for similar_user in similar_users:
            similar_user_movie_group = train_df.get_group(similar_user)
            similar_rating = similar_user_movie_group.loc[similar_user_movie_group[movie_col] == movie_id][
                rating_col]
            if len(similar_rating) > 0:
                n += 1
                ratings.append(similar_rating.iloc[0])
                if n >= N:
                    break

        assert len(ratings) == N
        predicted_rating = statistics.median(ratings)

        task_validation_df_copy[rating_col][ind] = predicted_rating

    current_error = sum([(expected - predicted) ** 2 for expected, predicted in
                         zip(validation_df[rating_col], task_validation_df_copy[rating_col])]) / len(validation_df)

    if current_error < error.value:
        with error.get_lock():
            if current_error >= error.value:
                return
            error.value = current_error
            final_n.value = N


def custom_error_callback(error):
    print(f'Got error: {error}')


if __name__ == '__main__':
    original_train_df = load_csv_to_df("train.csv")
    users = original_train_df[user_col].unique()
    train_df = original_train_df.groupby(user_col, group_keys=True)[[movie_col, rating_col]]

    manager = multiprocessing.Manager()
    user_to_similar_users = manager.dict()
    n_cores = 10
    pool = multiprocessing.Pool(n_cores)

    for user_id in users:
        pool.apply_async(get_similar_users,
                         args=(user_id, train_df, users, user_to_similar_users))

    pool.close()
    pool.join()

    validation_df = original_train_df.copy()
    task_validation_df = validation_df.copy().assign(rating=None)

    pool2 = multiprocessing.Pool(n_cores)
    error = multiprocessing.Value('f', math.inf)
    final_n = multiprocessing.Value('i', 0)
    # for N in range(1, 22, 2):
    #     pool2.apply_async(validate_n,
    #                       args=(task_validation_df, user_to_similar_users, train_df, N, error, final_n),
    #                       error_callback=custom_error_callback)
    #
    # pool2.close()
    # pool2.join()

    processes = [Process(target=validate_n,
                         args=(task_validation_df, validation_df, user_to_similar_users, train_df, N, error, final_n)) for N
                 in range(1, 20, 2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    # for N in range(1, 22, 2):
    #     task_validation_df_copy = task_validation_df.copy()
    #     for ind in task_validation_df_copy.index:
    #         user_id = task_validation_df_copy[user_col][ind]
    #         movie_id = task_validation_df_copy[movie_col][ind]
    #         similar_users = user_to_similar_users[user_id]
    #
    #         ratings = []
    #         n = 0
    #         for similar_user in similar_users:
    #             similar_user_movie_group = train_df.get_group(similar_user)
    #             similar_rating = similar_user_movie_group.loc[similar_user_movie_group[movie_col] == movie_id][
    #                 rating_col]
    #             if len(similar_rating) > 0:
    #                 n += 1
    #                 ratings.append(similar_rating.iloc[0])
    #                 if n >= N:
    #                     break
    #
    #         assert len(ratings) == N
    #         predicted_rating = statistics.median(ratings)
    #
    #         task_validation_df_copy[rating_col][ind] = predicted_rating
    #
    #     current_error = sum([(expected - predicted) ** 2 for expected, predicted in
    #                          zip(validation_df[rating_col], task_validation_df_copy[rating_col])]) / len(validation_df)
    #
    #     if current_error < error:
    #         error = current_error
    #         final_n = N

    N = final_n.value
    print("number of similar: " + str(N))
    task_df = load_csv_to_df("task.csv")
    task_df = task_df.assign(rating=None)

    for ind in task_df.index:
        user_id = task_df[user_col][ind]
        movie_id = task_df[movie_col][ind]
        similar_users = user_to_similar_users[user_id]

        ratings = []
        n = 0
        for similar_user in similar_users:
            similar_user_movie_group = train_df.get_group(similar_user)
            similar_rating = similar_user_movie_group.loc[similar_user_movie_group[movie_col] == movie_id][
                rating_col]
            if len(similar_rating) > 0:
                n += 1
                ratings.append(similar_rating.iloc[0])
                if n >= N:
                    break

        assert len(ratings) == N
        predicted_rating = statistics.median(ratings)

        task_df[rating_col][ind] = predicted_rating

    print(task_df)
    task_df.to_csv("submission.csv", sep=';', header=False)
