import math
import multiprocessing
from multiprocessing import Process
from multiprocessing import Lock

import sys
import os
import pandas as pd
import requests
from collections import Counter

# api docs: https://developers.themoviedb.org/3/movies/
api_key = os.environ['API_KEY']
api = 'https://api.themoviedb.org/3/movie/'

# train and task csv columns
index_col = "index"  # int
user_col = "user_id"  # int
movie_col = "movie_id"  # int
rating_col = "rating"  # int/NA(float), 0-5 inclusive


def load_csv_to_df(csv_path):
    # ordering of user_id and movie_id can be changed here
    df = pd.read_csv(csv_path, delimiter=";", index_col=0, names=[index_col, user_col, movie_col, rating_col])
    return df


def request_data(task_df, train_df, endpoint_to_response_cache, lock, data, endpoint="", unique_id="id"):
    if endpoint not in endpoint_to_response_cache:
        with lock:
            if endpoint in endpoint_to_response_cache:
                return {movie_id: set([entry[unique_id] for entry in response.get(data, [])]) for movie_id, response in
                        endpoint_to_response_cache[endpoint].items()}

            movie_to_response = {movie_id: None for movie_id in
                                 {movie_id for movie_id in task_df[movie_col].unique()} | {movie_id for movie_id in
                                                                                           train_df[
                                                                                               movie_col].unique()}}
            for movie_id in movie_to_response.keys():
                movie_to_response[movie_id] = requests.get(
                    f'{api}{movie_id}{endpoint}?api_key={api_key}').json()

            endpoint_to_response_cache[endpoint] = movie_to_response
    return {movie_id: set([entry[unique_id] for entry in response.get(data, [])]) for movie_id, response in
            endpoint_to_response_cache[endpoint].items()}


def get_movie_to_keywords(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "keywords", "/keywords")


def get_movie_to_genres(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "genres")


def get_movie_to_cast(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "cast", "/credits")


def get_movie_to_country(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "production_countries", "", "name")


def calculate_hamming_distance_of_features(features):
    # hamming distance with weights
    # change keyword movement with weight of 10, and add/remove keyword movement with weight of 1
    change_keyword_weight = 10
    add_remove_keyword_weight = 1
    return change_keyword_weight * abs(
        features[0] - min(features[1], features[2])) + add_remove_keyword_weight * abs(features[1] - features[2])


def calculate_euclidean_distance_of_neighbours(neighbour_features):
    return math.sqrt(sum(feature ** 2 for feature in neighbour_features[1:]))


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def knn(task_df, train_df, lock, endpoint_to_response_cache, k, keywords_weight, genres_weight, cast_weight,
        countries_weight):
    movie_to_keywords = get_movie_to_keywords(task_df, train_df, endpoint_to_response_cache, lock)
    movie_to_genres = get_movie_to_genres(task_df, train_df, endpoint_to_response_cache, lock)
    movie_to_cast = get_movie_to_cast(task_df, train_df, endpoint_to_response_cache, lock)
    movie_to_country = get_movie_to_country(task_df, train_df, endpoint_to_response_cache, lock)

    for task_i, task_row in task_df.iterrows():
        user_id = task_row[user_col]
        task_movie_id = task_row[movie_col]

        task_keywords = movie_to_keywords[task_movie_id]
        task_genres = movie_to_genres[task_movie_id]
        task_cast = movie_to_cast[task_movie_id]
        task_countries = movie_to_country[task_movie_id]

        neighbours = []
        min_hamming_distance_genres = math.inf
        max_hamming_distance_genres = 0
        min_hamming_distance_keywords = math.inf
        max_hamming_distance_keywords = 0
        min_hamming_distance_cast = math.inf
        max_hamming_distance_cast = 0
        min_hamming_distance_countries = math.inf
        max_hamming_distance_countries = 0
        for train_i, train_row in train_df.loc[train_df[user_col] == user_id].iterrows():
            train_movie_id = train_row[movie_col]
            train_rate = train_row[rating_col]

            train_keywords = movie_to_keywords[train_movie_id]
            keywords_covered = len(train_keywords.intersection(task_keywords))
            hamming_distance_keywords = calculate_hamming_distance_of_features(
                (keywords_covered, len(task_keywords), len(train_keywords)))
            min_hamming_distance_keywords = min(min_hamming_distance_keywords, hamming_distance_keywords)
            max_hamming_distance_keywords = max(max_hamming_distance_keywords, hamming_distance_keywords)

            train_genres = movie_to_genres[train_movie_id]
            genres_covered = len(train_genres.intersection(task_genres))
            hamming_distance_genres = calculate_hamming_distance_of_features(
                (genres_covered, len(task_genres), len(train_genres)))
            min_hamming_distance_genres = min(min_hamming_distance_genres, hamming_distance_genres)
            max_hamming_distance_genres = max(max_hamming_distance_genres, hamming_distance_genres)

            train_cast = movie_to_cast[train_movie_id]
            cast_covered = len(train_cast.intersection(task_cast))
            hamming_distance_cast = calculate_hamming_distance_of_features(
                (cast_covered, len(task_cast), len(train_cast)))
            min_hamming_distance_cast = min(min_hamming_distance_cast, hamming_distance_cast)
            max_hamming_distance_cast = max(max_hamming_distance_cast, hamming_distance_cast)

            train_countries = movie_to_country[train_movie_id]
            countries_covered = len(train_countries.intersection(task_countries))
            hamming_distance_countries = calculate_hamming_distance_of_features(
                (countries_covered, len(task_countries), len(train_countries)))
            min_hamming_distance_countries = min(min_hamming_distance_countries, hamming_distance_countries)
            max_hamming_distance_countries = max(max_hamming_distance_countries, hamming_distance_countries)

            neighbours.append((train_rate, hamming_distance_keywords, hamming_distance_genres, hamming_distance_cast,
                               hamming_distance_countries))

        # normalization and weights
        neighbours = [(neighbour[0],
                       10 ** keywords_weight * normalize(neighbour[1], min_hamming_distance_keywords,
                                                   max_hamming_distance_keywords),
                       10 ** genres_weight * normalize(neighbour[2], min_hamming_distance_genres,
                                                 max_hamming_distance_genres),
                       10 ** cast_weight * normalize(neighbour[3], min_hamming_distance_cast,
                                               max_hamming_distance_cast),
                       10 ** countries_weight * normalize(neighbour[4], min_hamming_distance_countries,
                                                    max_hamming_distance_countries)) for neighbour
                      in neighbours]

        k_nearest_neighbours_rates = [neighbour[0] for neighbour in
                                      sorted(neighbours, key=calculate_euclidean_distance_of_neighbours)[:k]]
        counter = Counter(k_nearest_neighbours_rates)
        most_frequent_rate = round(sum([rate * count for rate, count in counter.most_common()]) / sum(
            filter(lambda count: count > 0, counter.values())))
        assert task_df.loc[task_i, rating_col] is None, "must be NaN before assignment"
        task_df.loc[task_i, rating_col] = int(most_frequent_rate)
    task_df[rating_col] = task_df[rating_col].astype(int)


def cross_validate(task_validation_df, train_validation_df, k, endpoint_to_response_cache_lock,
                   endpoint_to_response_cache, min_error,
                   result, result_value, keywords_weight=0, genres_weight=0, cast_weight=0,
                   countries_weight=0):
    task_validation_df_copy = task_validation_df.copy()
    knn(task_validation_df_copy, train_validation_df, endpoint_to_response_cache_lock, endpoint_to_response_cache, k,
        keywords_weight=keywords_weight,
        genres_weight=genres_weight,
        cast_weight=cast_weight, countries_weight=countries_weight)
    # validation_df instead of train_validation_df used?
    error = sum([abs(expected - predicted) for expected, predicted in
                 zip(train_validation_df[rating_col], task_validation_df_copy[rating_col])])
    if error < min_error.value:
        with min_error.get_lock():
            if error >= min_error.value:
                return
            min_error.value = error
            result.value = result_value


if __name__ == '__main__':
    train_df = load_csv_to_df("train.csv")

    shuffled_train_df = train_df.sample(frac=1)
    validation_part = 0.25
    validation_end_row = int(validation_part * len(shuffled_train_df.index))
    validation_df = shuffled_train_df.iloc[:validation_end_row, :]
    task_validation_df = validation_df.copy().assign(rating=None)
    train_validation_df = shuffled_train_df.iloc[validation_end_row:, :]

    lock = Lock()
    manager = multiprocessing.Manager()
    endpoint_to_response_cache = manager.dict()
    min_error = multiprocessing.Value('f', math.inf)
    result = multiprocessing.Value('i', 0)

    processes = [Process(target=cross_validate,
                         args=(
                             task_validation_df, train_validation_df, k, lock, endpoint_to_response_cache, min_error,
                             result, k)) for k in
                 range(15, 35)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    neighbours_number = result.value
    print(f"k = {neighbours_number}")
    print(f"min_error = {min_error.value}")
    print()

    min_error.value = math.inf
    result.value = 0
    processes = [Process(target=cross_validate,
                         args=(
                             task_validation_df, train_validation_df, neighbours_number, lock,
                             endpoint_to_response_cache, min_error,
                             result, keywords_weight, keywords_weight)) for keywords_weight in
                 range(-9, 11)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    keywords_weight_final = result.value
    print(f"keywords_weight = {keywords_weight_final}")
    print(f"min_error = {min_error.value}")
    print()

    min_error.value = math.inf
    result.value = 0
    processes = [Process(target=cross_validate,
                         args=(
                             task_validation_df, train_validation_df, neighbours_number, lock,
                             endpoint_to_response_cache, min_error,
                             result, genres_weight, keywords_weight_final, genres_weight)) for genres_weight in
                 range(-9, 11)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    genres_weight_final = result.value
    print(f"genres_weight = {genres_weight_final}")
    print(f"min_error = {min_error.value}")
    print()

    min_error.value = math.inf
    result.value = 0
    processes = [Process(target=cross_validate,
                         args=(
                             task_validation_df, train_validation_df, neighbours_number, lock,
                             endpoint_to_response_cache, min_error,
                             result, cast_weight, keywords_weight_final, genres_weight_final, cast_weight)) for
                 cast_weight in
                 range(-9, 11)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    cast_weight_final = result.value
    print(f"cast_weight = {cast_weight_final}")
    print(f"min_error = {min_error.value}")
    print()

    min_error.value = math.inf
    result.value = 0
    processes = [Process(target=cross_validate,
                         args=(
                             task_validation_df, train_validation_df, neighbours_number, lock,
                             endpoint_to_response_cache, min_error,
                             result, countries_weight, keywords_weight_final, genres_weight_final, cast_weight_final,
                             countries_weight))
                 for countries_weight in
                 range(-9, 11)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    countries_weight_final = result.value
    print(f"countries_weight = {countries_weight_final}")
    print(f"min_error = {min_error.value}")
    print()

    task_df = load_csv_to_df("task.csv")
    task_df = task_df.assign(rating=None)

    knn(task_df, train_df, lock, endpoint_to_response_cache, neighbours_number, keywords_weight=keywords_weight_final,
        genres_weight=genres_weight_final,
        cast_weight=cast_weight_final, countries_weight=countries_weight_final)

    task_df.to_csv("submission.csv", sep=';', header=False)

    with open('features.txt', 'w') as sys.stdout:
        print(f"k = {neighbours_number}")
        print(f"keywords_weight = {keywords_weight_final}")
        print(f"genres_weight = {genres_weight_final}")
        print(f"cast_weight = {cast_weight_final}")
        print(f"countries_weight = {countries_weight_final}")
