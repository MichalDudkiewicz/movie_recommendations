import math
import multiprocessing
from multiprocessing import Process
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


# train and task csv columns
index_col = "index"  # int
user_col = "user_id"  # int
movie_col = "movie_id"  # int
rating_col = "rating"  # int/NA(float), 0-5 inclusive

def load_csv_to_df(csv_path):
    # ordering of user_id and movie_id can be changed here
    df = pd.read_csv(csv_path, delimiter=";", index_col=0, names=[index_col, user_col, movie_col, rating_col])
    return df


p_step = 0.1
x_step = 0.1
eps = 0.01

def validate(movies_to_persons, merged, min_error, result, n, train_validation_df, task_df):
    ones = np.ones((n + 1, len(movies_to_persons.columns)))
    parameters = pd.DataFrame(ones, columns=movies_to_persons.columns)

    ones = np.ones((len(movies_to_persons.index), n))
    features = pd.DataFrame(ones, index=movies_to_persons.index)

    feature_ids = features.columns.unique()
    movie_ids = merged[movie_col].unique()
    user_ids = merged[user_col].unique()

    user_ids_left = set(user_ids)
    movie_ids_left = set(movie_ids)

    while len(user_ids_left) > 0 or len(movie_ids_left) > 0:
        if len(user_ids_left) > 0:
            p_derivatives = [0] * len(parameters.index)
            user_id = user_ids_left.pop()
            watched_movies = 0
            for movie_id in movie_ids:
                rating = movies_to_persons.loc[movie_id, user_id]
                if not math.isnan(rating):
                    watched_movies += 1
                    parameter_id = 0
                    f = parameters.loc[parameter_id, user_id]
                    for feature_id in feature_ids:
                        parameter_id += 1
                        f += features.loc[movie_id, feature_id] * parameters.loc[parameter_id, user_id]
                    diff = f - rating
                    for derivative_id in range(len(p_derivatives)):
                        x = features.loc[movie_id, derivative_id - 1] if derivative_id > 0 else 1
                        p_derivatives[derivative_id] += diff * x

            p_derivatives = [p_div / watched_movies for p_div in p_derivatives]

            if all(abs(p_div) < eps for p_div in p_derivatives):
                continue
            user_ids_left.add(user_id)

            for derivative_id in range(len(p_derivatives)):
                parameters.loc[derivative_id, user_id] = parameters.loc[derivative_id, user_id] - p_step * \
                                                         p_derivatives[derivative_id]

        if len(movie_ids_left) > 0:
            movie_id = movie_ids_left.pop()
            x_derivatives = [0] * len(feature_ids)
            watched_by = 0
            for user_id in user_ids:
                rating = movies_to_persons.loc[movie_id, user_id]
                if not math.isnan(rating):
                    watched_by += 1
                    parameter_id = 0
                    f = parameters.loc[parameter_id, user_id]
                    for feature_id in feature_ids:
                        parameter_id += 1
                        f += features.loc[movie_id, feature_id] * parameters.loc[parameter_id, user_id]
                    diff = f - rating
                    for derivative_id in range(len(x_derivatives)):
                        p = parameters.loc[derivative_id + 1, user_id]
                        x_derivatives[derivative_id] += diff * p

            x_derivatives = [x_div / watched_by for x_div in x_derivatives]

            if all(abs(x_div) < eps for x_div in x_derivatives):
                continue
            movie_ids_left.add(movie_id)

            for derivative_id in range(len(x_derivatives)):
                features.loc[movie_id, derivative_id] = features.loc[movie_id, derivative_id] - x_step * x_derivatives[
                    derivative_id]

    for movie_id in movie_ids:
        for user_id in user_ids:
            rating = movies_to_persons.loc[movie_id, user_id]
            if math.isnan(rating):
                parameter_id = 0
                predicted_rating = parameters.loc[parameter_id, user_id]
                for feature_id in feature_ids:
                    parameter_id += 1
                    predicted_rating += features.loc[movie_id, feature_id] * parameters.loc[parameter_id, user_id]
                movies_to_persons.loc[movie_id, user_id] = round(predicted_rating)

    for ind in task_df.index:
        task_df[rating_col][ind] = int(movies_to_persons.loc[task_df[movie_col][ind], task_df[user_col][ind]])
    task_df[rating_col] = task_df[rating_col].astype(int)

    error = sum([abs(expected - predicted) for expected, predicted in
                 zip(train_validation_df[rating_col], task_df[rating_col])])
    if error < min_error.value:
        with min_error.get_lock():
            if error >= min_error.value:
                return
            min_error.value = error
            result.value = n


if __name__ == '__main__':
    train_df = load_csv_to_df("train.csv")
    task_df = load_csv_to_df("task.csv")
    task_df = task_df.assign(rating=pd.NA)

    min_error = multiprocessing.Value('f', math.inf)
    result = multiprocessing.Value('i', 0)

    train_df_copy = train_df.copy()
    validation_part = 0.25

    y = train_df_copy.pop(user_col).to_frame()
    train_validation_df, task_validation_df, y_train_validation_df, y_task_validation_df = train_test_split(
        train_df_copy, y, stratify=y, test_size=validation_part)

    train_validation_df = y_train_validation_df.join(train_validation_df)
    validation_df = y_task_validation_df.join(task_validation_df)
    task_validation_df = validation_df.copy().assign(rating=pd.NA)

    merged = pd.concat([train_validation_df, task_validation_df])
    movies_to_persons = merged.pivot_table(values=movie_col, index=movie_col, columns=user_col, aggfunc='first', dropna=False)

    processes = [Process(target=validate,
                         args=(movies_to_persons, merged, min_error, result, n, validation_df, task_validation_df)) for n in
                 range(1, 21)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    merged = pd.concat([train_df, task_df])
    movies_to_persons = merged.pivot_table(values=movie_col, index=movie_col, columns=user_col, aggfunc='first',
                                           dropna=False)

    n = result.value
    # wyszlo 3, a wczesniej testowalem 5
    print("n = " + str(n))

    ones = np.ones((n + 1, len(movies_to_persons.columns)))
    parameters = pd.DataFrame(ones, columns=movies_to_persons.columns)

    ones = np.ones((len(movies_to_persons.index), n))
    features = pd.DataFrame(ones, index=movies_to_persons.index)

    feature_ids = features.columns.unique()
    movie_ids = merged[movie_col].unique()
    user_ids = merged[user_col].unique()

    user_ids_left = set(user_ids)
    movie_ids_left = set(movie_ids)

    print("users left: " + str(len(user_ids_left)) + "/" + str(len(user_ids)), flush=True)
    print("movies left: " + str(len(movie_ids_left)) + "/" + str(len(movie_ids)))

    while len(user_ids_left) > 0 or len(movie_ids_left) > 0:
        if len(user_ids_left) > 0:
            p_derivatives = [0] * len(parameters.index)
            user_id = user_ids_left.pop()
            watched_movies = 0
            for movie_id in movie_ids:
                rating = movies_to_persons.loc[movie_id, user_id]
                if not math.isnan(rating):
                    watched_movies += 1
                    parameter_id = 0
                    f = parameters.loc[parameter_id, user_id]
                    for feature_id in feature_ids:
                        parameter_id += 1
                        f += features.loc[movie_id, feature_id] * parameters.loc[parameter_id, user_id]
                    diff = f - rating
                    for derivative_id in range(len(p_derivatives)):
                        x = features.loc[movie_id, derivative_id - 1] if derivative_id > 0 else 1
                        p_derivatives[derivative_id] += diff * x

            p_derivatives = [p_div/watched_movies for p_div in p_derivatives]

            if all(abs(p_div) < eps for p_div in p_derivatives):
                print("users left: " + str(len(user_ids_left)) + "/" + str(len(user_ids)), flush=True)
                print("movies left: " + str(len(movie_ids_left)) + "/" + str(len(movie_ids)))
                continue
            user_ids_left.add(user_id)

            for derivative_id in range(len(p_derivatives)):
                parameters.loc[derivative_id, user_id] = parameters.loc[derivative_id, user_id] - p_step * p_derivatives[derivative_id]

        if len(movie_ids_left) > 0:
            movie_id = movie_ids_left.pop()
            x_derivatives = [0] * len(feature_ids)
            watched_by = 0
            for user_id in user_ids:
                rating = movies_to_persons.loc[movie_id, user_id]
                if not math.isnan(rating):
                    watched_by += 1
                    parameter_id = 0
                    f = parameters.loc[parameter_id, user_id]
                    for feature_id in feature_ids:
                        parameter_id += 1
                        f += features.loc[movie_id, feature_id] * parameters.loc[parameter_id, user_id]
                    diff = f - rating
                    for derivative_id in range(len(x_derivatives)):
                        p = parameters.loc[derivative_id + 1, user_id]
                        x_derivatives[derivative_id] += diff * p

            x_derivatives = [x_div / watched_by for x_div in x_derivatives]

            if all(abs(x_div) < eps for x_div in x_derivatives):
                print("users left: " + str(len(user_ids_left)) + "/" + str(len(user_ids)), flush=True)
                print("movies left: " + str(len(movie_ids_left)) + "/" + str(len(movie_ids)))
                continue
            movie_ids_left.add(movie_id)

            for derivative_id in range(len(x_derivatives)):
                features.loc[movie_id, derivative_id] = features.loc[movie_id, derivative_id] - x_step * x_derivatives[derivative_id]

    for movie_id in movie_ids:
        for user_id in user_ids:
            rating = movies_to_persons.loc[movie_id, user_id]
            if math.isnan(rating):
                parameter_id = 0
                predicted_rating = parameters.loc[parameter_id, user_id]
                for feature_id in feature_ids:
                    parameter_id += 1
                    predicted_rating += features.loc[movie_id, feature_id] * parameters.loc[parameter_id, user_id]
                movies_to_persons.loc[movie_id, user_id] = round(predicted_rating)

    for ind in task_df.index:
        task_df[rating_col][ind] = int(movies_to_persons.loc[task_df[movie_col][ind], task_df[user_col][ind]])
    task_df[rating_col] = task_df[rating_col].astype(int)
    task_df.to_csv("submission.csv", sep=';', header=False)

