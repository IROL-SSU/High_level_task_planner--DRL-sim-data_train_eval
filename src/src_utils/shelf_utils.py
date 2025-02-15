import yaml
import numpy as np

# YAML 파일 로드 함수
def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Pose 데이터를 numpy 배열로 변환하고 정렬하는 함수
def get_sorted_pose_matrix(pose_dict):
    pose_list = []
    
    # (x, y) 좌표를 기준으로 pose 정렬
    for obj_name, pose in sorted(pose_dict.items(), key=lambda item: (item[1][0], item[1][1])):  
        pose_list.append(np.array(pose))  # 정렬된 Pose 추가
    
    # (N x 7) 형태의 numpy 배열 생성
    pose_array = np.vstack(pose_list)
    
    # 물체 개수에 맞춰 적절한 행렬 형태 찾기 (예: 3x2 또는 2x3)
    num_objects = len(pose_list)
    
    # 가능한 배치 형태 찾기
    for i in range(1, num_objects + 1):
        if num_objects % i == 0:  # 나누어 떨어지는 경우
            rows, cols = i, num_objects // i
            break

    # 배열 재구성
    pose_array = pose_array.reshape(rows, cols, -1)  # (rows, cols, 7) 형태

    return pose_array

if __name__ == "__main__":
    yaml_path = "src/shelf_policy/params/environment.yaml"
    config = load_yaml_config(yaml_path)
    object_dict = config["objects"]
    pose_dict = config["pose"]
    for obj_name, pose in pose_dict.items():
        print(np.array(pose, dtype=np.float32))
    # object_arrangement = get_sorted_pose_matrix(pose_dict)
    # print(object_arrangement)
