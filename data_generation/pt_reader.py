import pickle
import os
output = "/data/TrafficFormer_1.0/data_generation/pretrain_output/"
dataset_pt_path = os.path.join(output, "dataset.pt")

print("文件大小(MB):", os.path.getsize(dataset_pt_path) / (1024 * 1024))
with open(dataset_pt_path, "rb") as f:
        print("文件是可读的吧？？？:", f.read(20))
        f.seek(0)
        first = pickle.load(f)
        count = 1
        try:
            while True:
                pickle.load(f)
                count += 1
                first = pickle.load(f)
                # print(count, first)

        except EOFError:
            pass
# print("样本数:", count)
print("第一条样本:", first)