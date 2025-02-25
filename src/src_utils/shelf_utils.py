import yaml
import numpy as np
import torch 
# YAML 파일 로드 함수
def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Pose 데이터를 numpy 배열로 변환하고 정렬하는 함수
def load_and_reshape_pose(pose_dict):
    """
    Sort by (x, y) coordinates, and reshape into (1, rows, cols, 7).
    """

    # 2. Sort poses by (x, y) values
    sorted_poses = sorted(pose_dict.items(), key=lambda item: (-item[1][0], item[1][1]))

    # 3. Convert to numpy array
    pose_list = [np.array(pose, dtype=np.float32) for _, pose in sorted_poses]
    pose_array = np.array(pose_list)  # (num_objects, 7) shape

    # 4. Determine shape dynamically
    num_objects = pose_array.shape[0]  # Total number of objects
    num_rows = len(set([pose[0] for pose in pose_array]))  # Unique x values
    num_cols = num_objects // num_rows  # Compute number of columns

    # 5. Reshape into (1, rows, cols, 7)
    pose_array = pose_array.reshape(1, num_rows, num_cols, 7)

    return tuple(map(tuple, pose_array.tolist()))
if __name__ == "__main__":
    yaml_path = "src/shelf_policy/params/environment.yaml"
    config = load_yaml_config(yaml_path)
    object_dict = config["objects"]
    pose_dict = config["pose"]

    # for obj_name, pose in pose_dict.items():
    #     print(np.array(pose, dtype=np.float32))
    object_arrangement = load_and_reshape_pose(pose_dict)
    print(object_arrangement)
