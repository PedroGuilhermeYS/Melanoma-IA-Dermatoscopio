# 🔬 Melanoma-IA-Dermatoscópio

![Submissão Latinoware](https://img.shields.io/badge/Submiss%C3%A3o-Latinoware-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-CC--BY--NC_4.0-lightgrey?style=for-the-badge)

Um projeto de Inteligência Artificial focado na classificação binária de lesões cutâneas e detecção precoce de melanoma, desenvolvido com a poderosa arquitetura de Visão Computacional (Vision Transformer) **EVA-02**. 

Este repositório é parte de um projeto submetido ao evento **Latinoware**, visando demonstrar o enorme potencial de soluções Open Source de Deep Learning no auxílio à análise dermatológica.

> **⚠️ AVISO MÉDICO IMPORTANTE:** Este sistema foi desenvolvido estritamente como uma **ferramenta de pesquisa** (Research Tool). **NÃO possui validação clínica** e jamais deve ser utilizado para substituir um diagnóstico profissional. Consulte sempre um médico dermatologista.

---

## 📖 Sobre o Projeto

O sistema recebe imagens dermatoscópicas (imagens de manchas ou pintas da pele tiradas com equipamento específico) e retorna uma predição probabilística indicando se a lesão apresenta características **Malignas** ou **Benignas**. 

O modelo é baseado na arquitetura **EVA-02 Small**, treinada via Masked Image Modeling, e que passou por refinamento (fine-tuning) em bases de imagens dermatológicas curadas em alta qualidade, como os datasets *HAM10000*, *BCN20000* e coleções do *ISIC*.

---

## 🛠️ Instalação (Passo a Passo)

Siga as etapas abaixo para configurar o ambiente e executar o classificador na sua própria máquina.

### 1. Pré-requisitos
- Ter o [Python](https://www.python.org/downloads/) (recomendado versão 3.8 a 3.11) instalado no sistema.
- Recomendamos o uso de um ambiente virtual (como `venv` ou `conda`) para não conflitar com outras bibliotecas do seu PC.

### 2. Instalação das Dependências
Abra o terminal (ou prompt de comando) na pasta do seu projeto e instale o framework de IA **PyTorch** e as demais bibliotecas auxiliares:

```bash
# Se o seu PC possui Placa de Vídeo NVIDIA (CUDA), utilize este comando:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Em seguida, instale as demais bibliotecas gráficas, modelos e o cliente do Hugging Face:
pip install timm pillow numpy huggingface_hub
```
*(Nota: Se o seu computador rodar em macOS ou for Windows/Linux sem placa de vídeo, você pode instalar o PyTorch tradicional através do comando `pip install torch torchvision`).*

---

## 🚀 Como Utilizar (Infrerência)

Com tudo instalado, você pode utilizar o script abaixo para analisar uma foto de lesão. O modelo vai baixar automaticamente os pesos treinados (checkpoint) diretamente do Hub do Hugging Face em sua primeira execução.

Crie um arquivo chamado `predict.py` e coloque uma foto dermatoscópica chamada `lesao.jpg` na mesma pasta, depois rode o código:

```python
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
```

---

## 🛑 Limitações Conhecidas

Ao demonstrar a tecnologia em eventos, é sempre transparente listar as limitações da IA:
* **Viés de Tom de Pele:** A grande base do treinamento (HAM10000, ISIC) foi construída em cima de pacientes com peles mais claras. O modelo sofre de degradação de acurácia com tons de pele mais escuros.
* **Artefatos:** O modelo pode acabar classificando incorretamente imagens caso haja pêlos demais sobre a mancha, marcadores com tinta de caneta médica ou sombras estranhas, gerando falsos correlações.

---

## 👏 Créditos e Agradecimentos (Base do Projeto)

Este repositório e a apresentação no **Latinoware** foram desenvolvidos utilizando como base de inteligência o excepcional trabalho *Open Source* de **Fabian Wolz**. 

* **Repositório Original do classificador base:** [FaGit99/melanoma-classifier-eva02](https://github.com/FaGit99/melanoma-classifier-eva02)
* **Autor do Treinamento e IA:** Fabian Wolz.
* **Origem das Imagens (Treino original):** Arquivos ISIC (HAM10000 de Tschandl et al., BCN20000 de Combalia et al., e datasets ISIC 2018/2019).
* **Biblioteca de Visão:** `timm` por Ross Wightman.

Este projeto visa a disseminação da educação na tecnologia. Um agradecimento especial a todos os desenvolvedores independentes que tornam seus conhecimentos públicos.

---

## ⚖️ Licença

Assim como as bases de imagens (HAM10000, BCN20000) e o repositório formador, este projeto encontra-se sob a licença **CC-BY-NC 4.0** (Não-Comercial). A replicação, estudo e uso acadêmico/pessoal são bem-vindos.