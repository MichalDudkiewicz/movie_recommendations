import math
import multiprocessing
from multiprocessing import Lock
import pygraphviz as pgv
import pandas as pd
import requests
import os
import uuid
from sklearn.model_selection import train_test_split

api_key = os.environ['API_KEY']
api = 'https://api.themoviedb.org/3/movie/'

possible_ratings = set(range(6))
# train and task csv columns
index_col = "index"  # int
user_col = "user_id"  # int
movie_col = "movie_id"  # int
rating_col = "rating"  # int/NA(float), 0-5 inclusive


def load_csv_to_df(csv_path):
    # ordering of user_id and movie_id can be changed here
    df = pd.read_csv(csv_path, delimiter=";", index_col=0, names=[index_col, user_col, movie_col, rating_col])
    return df


def p(objects_in_node, total_objects):
    assert total_objects != 0
    return objects_in_node / total_objects


def gini_index_for_node(p_class_node):
    objects_total = sum(p_class_node)
    if objects_total == 0:
        return 0
    return 1 - sum([(p / objects_total) ** 2 for p in p_class_node])


def get_movie_to_keywords(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "keywords", "/keywords")


def get_movie_to_genres(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "genres")


def get_movie_to_cast(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "cast", "/credits")


def get_movie_to_country(task_df, train_df, endpoint_to_response_cache, lock):
    return request_data(task_df, train_df, endpoint_to_response_cache, lock, "production_countries")


def request_data(task_df, train_df, endpoint_to_response_cache, lock, data, endpoint="", unique_id="name"):
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


def q(filtered_movie_ids, parent_node_train_set):
    num_parent_objects = len(parent_node_train_set)
    left_objects = parent_node_train_set[parent_node_train_set[movie_col].isin(filtered_movie_ids)]
    num_left_objects = len(left_objects)
    right_objects = parent_node_train_set[~parent_node_train_set[movie_col].isin(left_objects[movie_col])]
    num_right_objects = num_parent_objects - num_left_objects
    assert num_right_objects == len(right_objects)

    parent_p_class = [len(parent_node_train_set[parent_node_train_set[rating_col] == rate]) for rate in
                      possible_ratings]
    left_p_class = [len(left_objects[left_objects[rating_col] == rate]) for rate in possible_ratings]
    right_p_class = [len(right_objects[right_objects[rating_col] == rate]) for rate in possible_ratings]

    return gini_index_for_node(parent_p_class) - (
            p(num_left_objects, num_parent_objects) * gini_index_for_node(left_p_class) + p(num_right_objects,
                                                                                            num_parent_objects) * gini_index_for_node(
        right_p_class)), left_objects, right_objects


def induct_tree(parent_train_set, G, possible_genres, movie_to_genres, possible_keywords, movie_to_keywords,
                possible_cast, movie_to_cast, possible_countries, movie_to_country, depth_left, parent_node_id="",
                left=False):
    current_best_score = -math.inf
    current_best_label = ""
    node_color = "black"
    for genre in possible_genres:
        filtered_movie_ids = [movie_id for movie_id in movie_to_genres.keys() if
                              genre in movie_to_genres[movie_id]]
        if len(filtered_movie_ids) > 0 and len(
                parent_train_set[parent_train_set[movie_col].isin(filtered_movie_ids)]) > 0:
            score, left_set, right_set = q(filtered_movie_ids, parent_train_set)
            if current_best_score < score:
                current_best_score = score
                best_left_set = left_set
                best_right_set = right_set
                current_best_label = genre
                node_color = "red"

    for keyword in possible_keywords:
        filtered_movie_ids = [movie_id for movie_id in movie_to_keywords.keys() if
                              keyword in movie_to_keywords[movie_id]]
        if len(filtered_movie_ids) > 0 and len(
                parent_train_set[parent_train_set[movie_col].isin(filtered_movie_ids)]) > 0:
            score, left_set, right_set = q(filtered_movie_ids, parent_train_set)
            if current_best_score < score:
                current_best_score = score
                best_left_set = left_set
                best_right_set = right_set
                current_best_label = keyword
                node_color = "blue"

    for cast in possible_cast:
        filtered_movie_ids = [movie_id for movie_id in movie_to_cast.keys() if
                              cast in movie_to_cast[movie_id]]
        if len(filtered_movie_ids) > 0 and len(
                parent_train_set[parent_train_set[movie_col].isin(filtered_movie_ids)]) > 0:
            score, left_set, right_set = q(filtered_movie_ids, parent_train_set)
            if current_best_score < score:
                current_best_score = score
                best_left_set = left_set
                best_right_set = right_set
                current_best_label = cast
                node_color = "green"

    for country in possible_countries:
        filtered_movie_ids = [movie_id for movie_id in movie_to_country.keys() if
                              country in movie_to_country[movie_id]]
        if len(filtered_movie_ids) > 0 and len(
                parent_train_set[parent_train_set[movie_col].isin(filtered_movie_ids)]) > 0:
            score, left_set, right_set = q(filtered_movie_ids, parent_train_set)
            if current_best_score < score:
                current_best_score = score
                best_left_set = left_set
                best_right_set = right_set
                current_best_label = country
                node_color = "yellow"

    label = "YES" if left else "NO"
    if current_best_score != -math.inf and (len(best_left_set) != 0 and len(best_right_set) != 0) and depth_left > 0:
        node_id = str(uuid.uuid4())
        G.add_node(node_id, label=current_best_label, color=node_color)
        if parent_node_id != "":
            G.add_edge(parent_node_id, node_id, label=label)
        depth_left -= 1
        induct_tree(best_left_set, G, possible_genres, movie_to_genres, possible_keywords, movie_to_keywords,
                    possible_cast, movie_to_cast, possible_countries, movie_to_country, depth_left,
                    node_id, True)
        induct_tree(best_right_set, G, possible_genres, movie_to_genres, possible_keywords, movie_to_keywords,
                    possible_cast, movie_to_cast, possible_countries, movie_to_country, depth_left,
                    node_id, False)
    else:
        mode = parent_train_set[rating_col].mode()
        if len(mode) % 2 != 0:
            category = int(mode.median())
        else:
            mea = mode.mean()
            prev_abs = float('inf')
            for cat in mode:
                if abs(cat - mea) < prev_abs:
                    category = int(cat)
                    prev_abs = abs(cat - mea)
        leaf_id = str(uuid.uuid4())
        G.add_node(leaf_id, color="black", shape="rect", label=category)
        G.add_edge(parent_node_id, leaf_id, label=label)


def create_graph(user_train_df, possible_genres,
                 movie_to_genres,
                 possible_keywords, movie_to_keywords, possible_cast, movie_to_cast, possible_countries,
                 movie_to_country, max_depth, user_to_graph, user_id):
    # print("creating graph for user " + str(user_id))
    G = pgv.AGraph(directed=True)
    induct_tree(user_train_df, G, possible_genres,
                movie_to_genres,
                possible_keywords, movie_to_keywords, possible_cast, movie_to_cast, possible_countries,
                movie_to_country, max_depth)
    user_to_graph[user_id] = G.string()
    # print("user " + str(user_id) + " done")


def fill_task_df(task_df, ind, user_to_graph, movie_to_genres, movie_to_keywords, movie_to_cast, movie_to_country):
    user_id = task_df[user_col][ind]
    movie_id = task_df[movie_col][ind]
    # print("filling movie " + str(movie_id) + " for user " + str(user_id))

    decision_graph = pgv.AGraph(directed=True).from_string(user_to_graph[user_id])
    next_node = decision_graph.nodes()[0]
    while True:
        color = next_node.attr["color"]
        label = next_node.attr["label"]

        if color == "black":
            rating = int(label)
            break

        neighbors = decision_graph.out_neighbors(next_node)
        assert len(neighbors) == 2, "!= 2 children"
        # YES
        left = neighbors[0]

        # NO
        right = neighbors[1]

        condition = False
        # genre
        if color == "red":
            if label in movie_to_genres[movie_id]:
                condition = True
        # cast
        elif color == "green":
            if label in movie_to_cast[movie_id]:
                condition = True
        # keyword
        elif color == "blue":
            if label in movie_to_keywords[movie_id]:
                condition = True
        # country
        elif color == "yellow":
            if label in movie_to_country[movie_id]:
                condition = True

        if condition:
            next_node = left
        else:
            next_node = right

    # task_df.loc[ind, rating_col] = rating
    # print("[done] filling movie " + str(movie_id) + " for user " + str(user_id))
    return ind, rating


def custom_error_callback(error):
    print(f'Got error: {error}')


task_df = load_csv_to_df("task.csv")
task_validation_df_copy = pd.DataFrame()


def custom_callback(indToresult):
    task_df.loc[indToresult[0], rating_col] = indToresult[1]


def custom_validation_callback(indToresult):
    task_validation_df_copy.loc[indToresult[0], rating_col] = indToresult[1]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_df = load_csv_to_df("train.csv")

    lock = Lock()
    manager = multiprocessing.Manager()
    endpoint_to_response_cache = manager.dict()
    # task_df = load_csv_to_df("task.csv")
    task_df = task_df.assign(rating=None)

    movie_to_keywords = get_movie_to_keywords(task_df, train_df, endpoint_to_response_cache, lock)
    movie_to_genres = get_movie_to_genres(task_df, train_df, endpoint_to_response_cache, lock)
    movie_to_cast = get_movie_to_cast(task_df, train_df, endpoint_to_response_cache, lock)
    movie_to_country = get_movie_to_country(task_df, train_df, endpoint_to_response_cache, lock)

    possible_genres = set([genre for genres in [genres for genres in movie_to_genres.values()] for genre in genres])
    possible_keywords = set(
        [keyword for keywords in [keywords for keywords in movie_to_keywords.values()] for keyword in keywords])
    possible_cast = set([cast for casts in [casts for casts in movie_to_cast.values()] for cast in casts])
    possible_countries = set(
        [country for countries in [countries for countries in movie_to_country.values()] for country in countries])

    user_to_graph = manager.dict()
    n_cores = 10
    # todo: tutaj ustaw depth
    max_depth = 1

    train_df_copy = train_df.copy()
    validation_part = 0.25
    # validation_end_row = int(validation_part * len(shuffled_train_df.index))
    # validation_df = shuffled_train_df.iloc[:validation_end_row, :]
    #
    # train_validation_df = shuffled_train_df.iloc[validation_end_row:, :]
    # filtered_user_ids = [user_id for user_id in validation_df[user_col].unique() if
    #                      user_id in train_validation_df[user_col].unique()]
    # task_validation_df = validation_df.copy().assign(rating=None)
    # task_validation_df = task_validation_df[task_validation_df[user_col].isin(filtered_user_ids)]

    y = train_df_copy.pop(user_col).to_frame()
    train_validation_df, task_validation_df, y_train_validation_df, y_task_validation_df = train_test_split(
        train_df_copy, y, stratify=y, test_size=validation_part)

    train_validation_df = y_train_validation_df.join(train_validation_df)
    validation_df = y_task_validation_df.join(task_validation_df)
    task_validation_df = validation_df.copy().assign(rating=None)

    prev_error = math.inf
    prev_user_to_graph = {}

    # for depth in range(1, 5):
    #     print("Depth: " + str(depth), flush=True)
    #     pool = multiprocessing.Pool(n_cores)
    #
    #     for user_id in train_validation_df[user_col].unique():
    #         pool.apply_async(create_graph,
    #                          args=(
    #                              train_validation_df[train_validation_df[user_col] == user_id][[movie_col, rating_col]],
    #                              possible_genres,
    #                              movie_to_genres,
    #                              possible_keywords, movie_to_keywords, possible_cast, movie_to_cast, possible_countries,
    #                              movie_to_country, depth, user_to_graph, user_id),
    #                          error_callback=custom_error_callback)
    #
    #     pool.close()
    #     pool.join()
    #
    #     print("graph created")
    #
    #     task_validation_df_copy = task_validation_df.copy()
    #
    #     pool2 = multiprocessing.Pool(n_cores)
    #     for ind in task_validation_df_copy.index:
    #         pool2.apply_async(fill_task_df,
    #                           args=(
    #                               task_validation_df_copy, ind, user_to_graph, movie_to_genres, movie_to_keywords,
    #                               movie_to_cast,
    #                               movie_to_country), error_callback=custom_error_callback,
    #                           callback=custom_validation_callback)
    #
    #     pool2.close()
    #     pool2.join()
    #
    #     error = sum([1 for expected, predicted in
    #                  zip(validation_df[rating_col], task_validation_df_copy[rating_col]) if expected != predicted])
    #
    #     if error < prev_error:
    #         max_depth = depth
    #         print("error: " + str(error))
    #         prev_error = error
    #         prev_user_to_graph = user_to_graph
    #
    #     user_to_graph.clear()

    print("Final Depth: " + str(max_depth), flush=True)

    pool = multiprocessing.Pool(n_cores)

    for user_id in train_df[user_col].unique():
        pool.apply_async(create_graph,
                         args=(train_df[train_df[user_col] == user_id][[movie_col, rating_col]], possible_genres,
                               movie_to_genres,
                               possible_keywords, movie_to_keywords, possible_cast, movie_to_cast, possible_countries,
                               movie_to_country, max_depth, user_to_graph, user_id),
                         error_callback=custom_error_callback)

    pool.close()
    pool.join()

    print("graph created")

    user_to_draw = 1641
    deserialized_user_to_graph = pgv.AGraph(directed=True).from_string(user_to_graph[user_to_draw])
    deserialized_user_to_graph.layout('dot')
    deserialized_user_to_graph.draw("graphs/" + str(user_to_draw) + ".png")

    pool2 = multiprocessing.Pool(n_cores)
    for ind in task_df.index:
        pool2.apply_async(fill_task_df,
                          args=(
                              task_df, ind, user_to_graph, movie_to_genres, movie_to_keywords,
                              movie_to_cast,
                              movie_to_country), error_callback=custom_error_callback, callback=custom_callback)

    pool2.close()
    pool2.join()

    task_df.to_csv("submission.csv", sep=';', header=False)
