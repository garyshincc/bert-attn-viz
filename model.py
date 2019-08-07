##### some idiot model

import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"

tokens = tokenizer.tokenize(text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    _, _, attn_data_list = model(tokens_tensor)

# shape is 12 x batch_size x 12 x max_seq_len x max_seq_len

print(type(attn_data_list))
print(len(attn_data_list))
print(attn_data_list[0].shape)

last_layer_attn = attn_data_list[0][-1]
for attn_type in range(len(last_layer_attn)):
	attn = last_layer_attn[attn_type]
	plt.figure()
	df = pd.DataFrame(attn.cpu().numpy(), columns=text.split() + [" "])
	print(df)
	sns.heatmap(df, annot=True)
plt.show()



# # confirm we were able to predict 'henson'
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# assert predicted_token == 'henson'