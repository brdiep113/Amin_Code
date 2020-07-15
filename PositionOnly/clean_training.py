import json
import numpy as np


def clean_json_points(data_list, path_to_dir):
    for i in range(len(data_list)):
        point_info = {"X" : [], "Y" : []}
        for building in data_list[i]["img_info"]:
            json_path = path_to_dir + "/{0}.json".format(i)
            point_info["X"].extend(building["building_info"]["X"])
            point_info["Y"].extend(building["building_info"]["Y"])
        with open(json_path, 'w') as outfile:
            json.dump(point_info, outfile)


def clean_json_labels(data_list, path_to_dir):
    for i in range(len(data_list)):
        label_info = []
        for building in data_list[i]["img_info"]:
            json_path = path_to_dir + "/{0}.json".format(i)
            label_info.extend(building["building_info"]["Coarse Labels"])
        with open(json_path, 'w') as outfile:
            json.dump(label_info, outfile)


"path/to/point_locations/x.json"
def read_points_json(path_to_file):
    with open(path_to_file) as json_file:
        data = json.load(json_file)
        x_pts = np.array((data["X"]))
        y_pts = np.array((data["Y"]))
        points_matrix = np.vstack((x_pts, y_pts)).T

    return points_matrix

"path/to/coarse_labels/x.json"
def read_coarse_labels_json(path_to_file):
    with open(path_to_file) as json_file:
        data = json.load(json_file)
        edge_matrix = np.array(data)

    return edge_matrix
