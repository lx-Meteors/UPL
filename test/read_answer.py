

file = "/mnt/zhaorunsong/repository/UPL/experiment/local_experiment/ICAE_Llama-3.2-1B_UPL/output/instruction_inference_results.json"

import json

# 假设你的原始文件是 data.json
with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 只保留指定字段
filtered_data = []
for item in data:
    filtered_item = {
        "subset": item.get("subset"),
        "context": item.get("context"),
        "question": item.get("question"),
        "answers": item.get("answers"),
        "generate": item.get("generate"),
        "rouge-f1": item.get("rouge-f1")
    }
    filtered_data.append(filtered_item)

# 保存处理后的数据
with open("/mnt/zhaorunsong/lx/UPL/experiment/500x_1B_UPL/filtered_data.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print("✅ 处理完成，输出文件为 filtered_data.json")
