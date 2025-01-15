import os
import torch


data_path = "./emo_space/"
emotion = ["amusement", "excitement", "awe", "contentment", "fear", "disgust", "anger", "sadness"]
data = []
total_num = []
for emo in emotion:
    path = os.path.join(data_path, emo)
    pt_data = [torch.load(os.path.join(path, d)).detach().cpu().unsqueeze(0) for d in os.listdir(path) if d.endswith("v2.pt")]
    num = len(pt_data)
    total_num.append(num)
    data.extend(pt_data)

start = 0
data_path = "./emo_space/property"
for i in range(8):
    end = start + total_num[i]
    tmp = torch.concatenate(data[start:end], axis=0)
    mean = torch.mean(tmp, dim=0)
    std = torch.std(tmp, dim=0)
    torch.save(mean, os.path.join(data_path, f"{emotion[i]}_mean_v2.pt"))
    torch.save(std, os.path.join(data_path, f"{emotion[i]}_std_v2.pt"))
    start = end
print("finish")

# for i in range(8):
#     end = start + total_num[i]
#     tmp = torch.concatenate(data[start:end], axis=0)
#     mean = torch.mean(tmp, dim=0)
#     cov = torch.cov(tmp.t())
#     torch.save(mean, os.path.join(data_path, f"{emotion[i]}_mean_multivariate.pt"))
#     torch.save(cov, os.path.join(data_path, f"{emotion[i]}_var_multivariate.pt"))
#     start = end
# print("finish")