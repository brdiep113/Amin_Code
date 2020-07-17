from utils.plot_functions import fill_scene, plot_sample_img, plot_ground_truth
from utils.clean_training import clean_json_labels, clean_json_points
from Structure.Scene import Scene
import json

# path_to_training_images = 'E:\Amin_Codes\Synthesized_Data\Dataset_Sample\Image'

path_to_training_images1 = '../Position_feature/Image'
path_to_training_images2 = '../PositionOnly/Image'

path_to_training_truth1 = '../Position_feature/BuildingSegment'
path_to_training_truth2 = '../PositionOnly/BuildingSegment'

path_to_training_json = '../Generate_Data/jsons/training.json'


path_to_testing_images1 = '../Position_feature/Testing/Image'
path_to_testing_images2 = '../PositionOnly/Testing/Image'

path_to_testing_truth1 = '../Position_feature/Testing/BuildingSegment'
path_to_testing_truth2 = '../PositionOnly/Testing/BuildingSegment'

path_to_testing_json = '../Generate_Data/jsons/testing.json'

training_json = []
validation_json = []
testing_json = []

noise = False

training_sample_size = 100
testing_sample_size = training_sample_size // 10

scene_size = (128, 128)

for i in range(training_sample_size):
    curr_scene_id = "{0}".format(i)
    curr_scene_id = curr_scene_id.zfill(10)
    scene = Scene(scene_size, curr_scene_id)
    max_buildings = 12
    fill_scene(max_buildings, scene, training_json)
    plot_sample_img(scene, path_to_training_images1, i, add_noise=noise)
    plot_sample_img(scene, path_to_training_images2, i, add_noise=noise)
    plot_ground_truth(scene, path_to_training_truth1, i)
    plot_ground_truth(scene, path_to_training_truth2, i)

with open(path_to_training_json, 'w') as outfile:
    json.dump(training_json, outfile)

clean_json_points(training_json, '../Position_feature/Point_Location')
clean_json_labels(training_json, '../Position_feature/Coarse_Label')
clean_json_points(training_json, '../PositionOnly/Point_Location')
clean_json_labels(training_json, '../PositionOnly/Coarse_Label')

for i in range(testing_sample_size):
    curr_scene_id = "{0}".format(i)
    scene = Scene(scene_size, curr_scene_id)
    curr_scene_id = curr_scene_id.zfill(10)
    max_buildings = 12
    fill_scene(max_buildings, scene, testing_json)
    plot_sample_img(scene, path_to_testing_images1, i, add_noise=noise)
    plot_sample_img(scene, path_to_testing_images2, i, add_noise=noise)
    plot_ground_truth(scene, path_to_testing_truth1, i)
    plot_ground_truth(scene, path_to_testing_truth2, i)

clean_json_points(testing_json, '../Position_feature/Testing/Point_Location')
clean_json_labels(testing_json, '../Position_feature/Testing/Coarse_Label')
clean_json_points(testing_json, '../PositionOnly/Testing/Point_Location')
clean_json_labels(testing_json, '../PositionOnly/Testing/Coarse_Label')

with open(path_to_testing_json, 'w') as outfile:
    json.dump(testing_json, outfile)

# for i in range(testing_sample_size):
#    curr_scene_id = "test_img{0}".format(i)
#    scene = Scene(scene_size, curr_scene_id)
#    max_buildings = 24
#    fill_scene(max_buildings, scene, testing_json)
#    plot_sample_img(scene, path_to_testing_images, i)
#    #plot_ground_truth(scene, path_to_testing_truth, i)

#with open(path_to_testing_json, 'w') as outfile:
#    json.dump(training_json, outfile)