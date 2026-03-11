import torch

# 1. 검증할 pth 파일 불러오기 (경고 방지용 weights_only=True 적용)
file_path = "khy_omok_model_ep7000.pth"
state_dict = torch.load(file_path, map_location="cpu", weights_only=True)

print(f"=== 🧠 {file_path} 가중치 검사 ===")

# 2. 딕셔너리 안의 계층(Layer) 이름과 숫자(Tensor) 확인
for layer_name, weight_tensor in state_dict.items():
    # 텐서의 평균값과 총합을 계산하여 출력
    mean_val = weight_tensor.float().mean().item()
    sum_val = weight_tensor.float().sum().item()
    
    print(f"🔹 계층: {layer_name}")
    print(f"   - 가중치 평균: {mean_val:.6f} | 총합: {sum_val:.6f}")
    
    # 너무 길어지니 첫 번째 계층(보통 conv1.weight)만 보고 멈춤
    break