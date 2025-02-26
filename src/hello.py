import torch

# 데이터 예제
A = torch.tensor([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
])

B = torch.tensor([[0], [1], [0]])  # 선택할 인덱스

# 배치 인덱싱 적용
selected_data = A[torch.arange(A.shape[0]), B.squeeze(-1)]

print(B.squeeze(-1))