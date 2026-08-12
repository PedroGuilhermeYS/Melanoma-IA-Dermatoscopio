import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from huggingface_hub import hf_hub_download

# 1. Download/Acesso ao arquivo do Modelo treinado
print("Baixando modelo...")
ckpt_path = hf_hub_download(repo_id="fawo/eva02-small-melanoma-classifier", filename="model_0001.pt")
MODEL_NAME = "eva02_small_patch14_336.mim_in22k_ft_in1k"

# 2. Definição da Arquitetura
class ISICModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, drop_path_rate=0.1)
        self.model.head = nn.Linear(self.model.head.in_features, 1) # Saída Binária (Maligno/Benigno)
        
    def forward(self, x):
        return self.model(x)

# 3. Preparando para execução (via CPU ou Placa de Vídeo)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ISICModel(MODEL_NAME)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# 4. Ajustes na Imagem
transform = create_transform(**resolve_data_config({}, model=model.model), is_training=False)
img_path = "lesao.jpg" # Altere para o nome da sua imagem de teste
img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

# 5. Classificando
print("Analisando a lesão...")
with torch.no_grad():
    prob = torch.sigmoid(model(img)).item()
    
# Limiar utilizado (0.430) que entrega ~97% de sensibilidade segundo estudos originais
label = "MALIGNANT (ALERTA)" if prob >= 0.430 else "benign (Benigno)"

print(f"\nResultado da Análise:")
print(f"Probabilidade: {prob:.4f}")
print(f"Classificação: {label}")