#R-learner DL:
import torch
import torch.nn as nn
from transformers import RobertaModel
from torch.utils.data import DataLoader, Dataset, TensorDataset


class OrthogonalRoBERTa(nn.Module):
    def __init__(self, model_name = 'roberta-base', config = None):
        super(OrthogonalRoBERTa, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size
        dims = config.get('hidden_dims', [512, 256])
        #defining the three heads:
        self.heads = nn.ModuleDict({
            'm_head': self.MLP(d_model, hidden_list1, out_dim = 2),
            'e_head': self.MLP(d_model, hidden_list2, out_dim = 1),
            'tau_head': self.MLP(d_model, dims, out_dim = 1)
        })
    def MLP(self, input_dim, hidden_dims_list, out_dim):
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims_list:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)
    def forward(self, input_ids, attention_mask):
        outputs = self.MLP(input_ids = input_ids, attention_mask = attention_mask)
        z = outputs.last_hidden_state[:, 0, :]
        m_logits = self.heads['m_head'](z)
        e_probs = self.sigmoid(self.heads['e_head'](z))
        tau_val = self.heads['tau_head'](z)
        return m_logits, e_prob, tau_val
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0.0, std = 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def orthogonal_r_loss(y_true, t_true, m_hat, e_hat, tau_hat, ortho_reg = 0.25):
    y_res = y_true - m_hat
    t_res = t_true - e_hat 
    standard_r_loss = torch.mean((y_res - t_res * tau_hat) ** 2)
    #orthogonalization term:
    epsilon = y_res - t_res * tau_hat
    epsilon_centered = epsilon - torch.mean(epsilon)
    t_res_centered = t_res - torch.mean(t_res)
    ortho_penaty = torch.mean((epsilon_centered * t_res_centered) ** 2)
    ortho_term = ortho_reg * ortho_penalty
    total_loss = ortho_term + standard_r_loss
    return total_loss, ortho_term, standard_r_loss






config = {'hidden_dims': [128, 32]}
hidden_dims_list = [128, 32]


optimizer = AdamW([
  {'params': model.encoder.parameters(), 'lr': 2e-5},
  {'params': model.heads.parameters(), 'lr': 5e-5}
])

def tokenize_for_gpt(ds, block_size = 128):
    def tokenize_fn(examples):
        texts = [str(x) for x in examples['text']]
        return tokenizer(texts, truncation = False)
    tokenzied = ds.map(tokenize_fn, batched = True, remove_columns = ds.column_names)
    def group_texts(samples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        total_length = (total_length // block_size) * block_size
        result = {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
        }
        result['labels'] = [x[:] for x in result['input_ids']]
        return result
    lm_ds = tokenized.map(group_texts, batched = True)
    lm_ds.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])
    return lm_ds


train_ds = tokenizer_for_gpts(train_corpus, block_size = 128)
test_ds = tokenize_for_gpt(test_corpus, block_size = 128)
train_ds.save_to_disk("data/val_corpus_arrow")


def build_gpt_loaders(block_size = 128, batch_size = 32):
    train_corpus, test_corpus = build_all_text_corpora()
    train_ds = tokenize_for_gpt(train_corpus, block_size = block_size)
    test_ds = tokenize_for_gpt(test_corpus, block_size = block_size)
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = causal_lm_collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = causal_lm_collate_fn)
    return train_loader, test_loader

def create_dataloader(X_exist, Y_exist, batch_size = 16):
    
#Finding the next 

#loading in the dataset:


def data_process(csv_df, ctr_col, trt_col, response_col):
    df = pd.read_csv(csv_df)
    data_list = []
    for idx, row in df.iterrows():
        data_list.append({
            'text': str(row[ctr_col]),
            'T': 0.0,
            'Y': float(row[response_col])
        })
        data_list.append({
            'text': str(row[trt_col]),
            'T': 1.0,
            'Y': float(row[response_col])
        })
    df_output = pd.DataFrame(data_list)
    return df_output

class OrthogonalRoBERTa(nn.Module):
    def __init__(self, model_name = 'distilroberta-base', config = None):
        super(OrthogonalRoBERTa, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size
        h_dims = config.get('hidden_dims') if config else [128, 32]
        self.heads = nn.ModuleDict({
            'm_head': self.MLP(d_model, h_dims, out_dims = 1), #m(X)
            'e_head': self.MLP(d_model, h_dims, out_dims = 1),
            'tau_head': self.MLP(d_model, h_dims, out_dims = 1)
        })
    def MLP(self, input_dim, hidden_dims_list, out_dim):
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims_list:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        z = outputs.last_hidden_state[:, 0, :]
        m_val = self.heads['m_head'](z).squeeze(-1)
        e_prob = torch.sigmoid(self.heads['e_head'](z)).squeeze(-1)
        tau_val = self.heads['tau_head'](z).squeeze(-1)
        return m_val, e_prob, tau_val
    def initialize_weights(self):
        for name, module in self.heads.named_modules():
            nn.init.kaiming_normal_(module.weight, mode = 'fan_out',
                nonlinearity = 'relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        #####
        last_linear = self.head['tau_head'][-1]
        nn.init.normal_(last_linear.weight, mean = 0.0, std = 0.01)


#creating a Dataset Objective here:
class SarcasmDataset(Dataset):
    def __init__(self, texts, T, Y, tokenizer, max_len = 64):
        self.texts = texts
        self.T = T
        self.Y = Y
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens = True,
            max_length = self.max_len, padding = 'max_length',
            truncation = True, return_tensor = 'pt'
        )
        return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'T': torch.tensor(self.T[idx], dtype = torch.float),
        'Y': torch.tensor(self.Y[idx], dtype = torch.float)
        }
    #Formalize the data as the input of the tokenizer/Sarcasm datasets:

#Creating another 
    
df = data_process('train.En.csv', ctr_col = 'tweet', trt_col = 'rephrase',
    response_col = 'sarcasm').sample(1000, random_state = 20).reset_index(drop = True)
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
dataset = SarcasmDataset(df['text'].values, df['T'].values, df['Y'].values, tokenizer)
dataloader = DataLoader(daatset, batch_size = 16, shuffle = True)
model = OrthogonalRoBERTa(model_name = 'distilroberta-base')
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
model.train()

for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        m_hat, e_hat, tau_hat = model.forward(
            batch['input_ids'], batch['attention_mask']
            )
        loss, ortho_loss, r_loss = orthogonal_r_loss(
            batch['Y'], batch['T'],
            m_hat, e_hat, tau_hat)
        loss.backward()
        optimizer.step()
    print(f"Final Batch Loss: {loss.item():.4f}(R-loss: {r_loss.item():.4f}, O-loss: {ortho_loss.item():.4f})")


y_tilde_hat = []
e_tilde_hat = []
tau_hat_list = []
model.eval()
with torch.no_grad():
    for batch in dataloader:
        m_hat, e_hat, tau_hat = model.forward(
            batch['input_ids'], batch['attention_mask']
        )
        y_tilde_hat.extend((batch['Y'] - m_hat).detach().numpy())
        e_tilde_hat.extend((batch['T'] - e_hat).detach().numpy())
        tau_hat_list.extend(tau_hat.detach().numpy())


























