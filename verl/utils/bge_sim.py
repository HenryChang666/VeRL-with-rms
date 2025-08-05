from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, jsonify
app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhanghaoxing/model_hub/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/zhanghaoxing/model_hub/bge-large-zh-v1.5').to("cuda")
model.eval()


# 语义相似度
def compute_similarity(sent1, sent2):
    encoded_input = tokenizer([sent1, sent2], padding=True, truncation=True, return_tensors='pt').to(model.device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sim = (sentence_embeddings[0]*sentence_embeddings[1]).sum().cpu().item()
    if sim <= 0.65:
        return 0.1
    elif sim >= 0.85:
        return 1.0
    else:
        # 0.65=>0.1; 0.70=>0.158; 0.75=>0.251; 0.80=>0.398; 0.85=>0.630; 0.90=>1
        return (1 / 3981.0717055349733 * 10000 ** sim - 0.1) * 1.69811 + 0.1
    return sim

@app.route('/get_bge_sim', methods=['POST'])
def get_bge_sim():
    data = request.json
    sent1 = data.get('sent1')
    sent2 = data.get('sent2')
    return jsonify({"sim": compute_similarity(sent1, sent2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5032)