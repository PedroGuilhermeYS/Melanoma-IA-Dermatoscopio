# 🔬 Melanoma-IA-Dermatoscópio
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-CC--BY--NC_4.0-lightgrey?style=for-the-badge)

Um projeto de Inteligência Artificial focado na classificação binária de lesões cutâneas e detecção precoce de melanoma, desenvolvido com a poderosa arquitetura de Visão Computacional (Vision Transformer) **EVA-02**. 

Este repositório é parte de um projeto submetido ao evento **Latinoware**, visando demonstrar o enorme potencial de soluções Open Source de Deep Learning no auxílio à análise dermatológica, integrando software avançado com hardware acessível.

> **⚠️ AVISO MÉDICO IMPORTANTE:** Este sistema foi desenvolvido estritamente como uma **ferramenta de pesquisa** (Research Tool). **NÃO possui validação clínica** e jamais deve ser utilizado para substituir um diagnóstico profissional. Consulte sempre um médico dermatologista.

> Ver [`NOTICE.md`](./NOTICE.md) para a distinção entre o trabalho original e o código deste repositório.

---

# Arquitetura

```
┌──────────────────────────┐   HTTP GET /snapshot    ┌──────────────────────────────┐
│ raspberry-pi-camera-web   │ ───────────────────────▶│ Melanoma-IA-Dermatoscopio     │
│ (repositório da câmera)   │                          │ (este repositório)            │
└──────────────────────────┘                          │                                │
                                                        │  melanoma_ia/                 │
      Cliente HTTP direto ──────────────────┐          │  ├── model.py       (arquitetura + checkpoint)
      (curl, Postman, outro app)            │          │  ├── inference.py   (pré-processamento + predição)
                                             ▼          │  ├── gradcam.py     (Grad-CAM, compartilhado)
                                    POST /predict        │  ├── api/           (FastAPI: servidor)
                                    POST /predict/gradcam │  ├── cli/predict.py (linha de comando)
                                    GET  /health          │  ├── pipeline/orchestrator.py (polling automático)
                                                        │  └── gui/gradcam_viewer.py (PyQt6, pesquisa)
                                                        └──────────────────────────────┘
```

---

## Estrutura do pacote

```
melanoma_ia/
├── config.py       # ModelSettings / APISettings / OrchestratorSettings, via ambiente
├── model.py          # ISICModel + load_checkpoint 
├── inference.py        # InferenceEngine: pré-processamento + predict()
├── gradcam.py             # Grad-CAM: GradCAMHooks, compute_gradcam, overlay_heatmap
├── schemas.py               # PredictionResult / GradCAMResult / HealthStatus
├── clients/camera_client.py  # cliente HTTP fino para o /snapshot da câmera
├── api/
│   ├── main.py                # FastAPI app factory 
│   └── routes.py                # POST /predict, POST /predict/gradcam, GET /health
├── cli/predict.py                 # CLI real para classificar uma imagem
├── pipeline/orchestrator.py         # sem SFTP, sem credenciais fixas
└── gui/gradcam_viewer.py              # GUI PyQt6 de pesquisa
```

---

## 📖 Sobre o Projeto

O sistema recebe imagens dermatoscópicas (imagens de manchas ou pintas da pele tiradas com equipamento específico) e retorna uma predição probabilística indicando se a lesão apresenta características **Malignas** ou **Benignas**. 

O modelo é baseado na arquitetura **EVA-02 Small**, treinada via Masked Image Modeling, e que passou por refinamento (fine-tuning) em bases de imagens dermatológicas curadas em alta qualidade, como os datasets *HAM10000*, *BCN20000* e coleções do *ISIC*.

---

## 🛠️ Instalação (Passo a Passo)

Siga as etapas abaixo para configurar o ambiente e executar o classificador na sua própria máquina ou dispositivo embarcado.

### 1. Pré-requisitos
- Ter o [Python](https://www.python.org/downloads/) (recomendado versão 3.8 a 3.11) instalado no sistema.
- Recomendamos o uso de um ambiente virtual (como `venv` ou `conda`) para não conflitar com outras bibliotecas.

### 2. Instalação das Dependências
Abra o terminal (ou prompt de comando) na pasta do seu projeto e instale o framework de IA **PyTorch** e as demais bibliotecas auxiliares:

```bash
# Em seguida, instale as demais bibliotecas gráficas, modelos e dependências:
cd Melanoma-IA-Dermatoscopio
python3.11 -m venv .venv
source .venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu      # CPU

pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128     # GPU NVIDIA (CUDA 12.8)

# demais dependências, de acordo com o que você vai usar:
pip install -r requirements-api.txt
# ou, para a GUI de pesquisa:
pip install -r requirements-gui.txt
# ou, para desenvolvimento/testes:
pip install -r requirements-dev.txt

cp .env.example .env
```
---

### Reprodutibilidade

Por padrão, MELANOMA_HF_REVISION está vazio e o download usa o branch padrão do repositório no Hugging Face Hub. Se o checkpoint publicado mudar, os resultados passados deixam de ser replicáveis. Antes de qualquer execução cujos resultados você pretende citar, defina no `.env`:

```bash
MELANOMA_HF_REVISION=<commit_sha_ou_tag>
```

O CSV results/threshold_key_indicators.csv documenta os pontos de operação candidatos e seus respectivos thresholds. MELANOMA_THRESHOLD=0.430 (padrão) corresponde a ~97% de sensibilidade.

---

### Imagem de amostra

Para obter uma imagem pública de teste:

```bash
python scripts/download_sample_image.py
```

---

## Como Utilizar
O projeto oferece quatro formas de uso, dependendo da sua necessidade: linha de comando para uma imagem isolada, um servidor HTTP para qualquer cliente consumir, um orquestrador para captura contínua a partir da Raspberry Pi, e uma GUI de pesquisa para inspeção visual com Grad-CAM.

### CLI

```bash
python -m melanoma_ia.cli.predict --image samples/lesao.jpg
python -m melanoma_ia.cli.predict --image samples/lesao.jpg --json
python -m melanoma_ia.cli.predict --image samples/lesao.jpg --checkpoint /caminho/para/epoch_10.pt
```

### API (servidor de inferência)
Qualquer cliente (curl, Postman, o orquestrador, ou outro serviço) pode chamar:

```bash
uvicorn melanoma_ia.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/lesao.jpg"
# {"probability": 0.1234, "label": "benign", "threshold": 0.43, "model_name": "...", "model_revision": "..."}

curl -X POST http://localhost:8000/predict/gradcam \
  -F "file=@samples/lesao.jpg"
# {"prediction": {...}, "heatmap_png_base64": "..."}

curl http://localhost:8000/health
```

Se MELANOMA_API_TOKEN estiver definido no .env, /predict e /predict/gradcam exigem Authorization: Bearer <token> (o /health continua aberto, para healthchecks do Docker/systemd).

### Docker

```bash
docker compose up api
docker compose --profile orchestrator up
```

Para GPU, troque dockerfile: Dockerfile por dockerfile: Dockerfile.gpu no docker-compose.yml e adicione --gpus all / a seção deploy.resources correspondente.

### Integração com a Raspberry Pi (orquestrador)

Configure MELANOMA_CAMERA_URL no .env apontando para o repositório raspberry-pi-camera-web já em execução, então:

```bash
# no .env, aponte para o repositório raspberry-pi-camera-web já em execução:
MELANOMA_CAMERA_URL=http://192.168.1.50:5000
MELANOMA_POLL_INTERVAL_SECONDS=2
MELANOMA_BATCH_SIZE=10
```

```bash
python -m melanoma_ia.pipeline.orchestrator
```

Cada frame classificado é anexado como uma linha JSON em results/orchestrator_log.jsonl (configurável via MELANOMA_RESULTS_PATH):

```json
{"frame_index": 1, "processed_at": 1699999999.1, "capture_timestamp": 1699999998.9, "probability": 0.12, "label": "benign", "threshold": 0.43, "model_name": "...", "model_revision": "..."}
```

Carregável direto com pandas.read_json(path, lines=True) para análise.

### GUI de pesquisa (Grad-CAM)

```bash
python -m melanoma_ia.gui.gradcam_viewer
```

Ferramenta interativa para inspecionar Grad-CAM, ajustar colormap/threshold de ativação, revisar resultados em lote e consultar metadados do ISIC Archive.

## Testes

```bash
pip install -r requirements-dev.txt
pytest -q
```

Os testes não baixam o checkpoint real do Hugging Face Hub: usam um ViT pequeno (`vit_tiny_patch16_224`, `pretrained=False`) instanciado localmente através do mesmo ISICModel/load_checkpoint que o código de produção usa.

## Variáveis de ambiente

Veja `.env.example` para a lista completa. Resumo:

| Variável | Descrição |
|---|---|
| `MELANOMA_MODEL_NAME` | Nome do backbone timm |
| `MELANOMA_HF_REPO_ID` / `MELANOMA_HF_FILENAME` / `MELANOMA_HF_REVISION` | Origem do checkpoint no Hugging Face Hub — **pine a revisão antes de publicar resultados** |
| `MELANOMA_THRESHOLD` | Threshold de decisão (padrão: 0.430, ~97% sensibilidade) |
| `MELANOMA_DEVICE` | `auto` / `cpu` / `cuda` |
| `MELANOMA_API_HOST` / `MELANOMA_API_PORT` / `MELANOMA_API_TOKEN` | Configuração do servidor FastAPI |
| `MELANOMA_CAMERA_URL` / `MELANOMA_CAMERA_TOKEN` | Onde o orquestrador encontra o repositório da câmera |
| `MELANOMA_POLL_INTERVAL_SECONDS` / `MELANOMA_BATCH_SIZE` / `MELANOMA_RESULTS_PATH` | Comportamento do orquestrador |

---

## 🛑 Limitações Conhecidas

* **Viés de Tom de Pele:** A grande base do treinamento (HAM10000, ISIC) foi construída em cima de pacientes com peles mais claras. O modelo sofre de degradação de acurácia com tons de pele mais escuros.
* **Artefatos:** O modelo pode acabar classificando incorretamente imagens caso haja pêlos demais sobre a mancha, marcadores com tinta de caneta médica ou sombras estranhas, gerando falsas correlações.

---

## 👏 Créditos e Agradecimentos (Base do Projeto)

Este repositório e a apresentação no **Latinoware** foram desenvolvidos utilizando como base de inteligência o excepcional trabalho *Open Source* de **Fabian Wolz**. 

* **Repositório Original do classificador base:** [FaGit99/melanoma-classifier-eva02](https://github.com/FaGit99/melanoma-classifier-eva02)
* **Autor do Treinamento e IA:** Fabian Wolz.
* **Origem das Imagens (Treino original):** Arquivos ISIC (HAM10000 de Tschandl et al., BCN20000 de Combalia et al., e datasets ISIC 2018/2019).
* **Biblioteca de Visão:** `timm` por Ross Wightman.

Este projeto visa a disseminação da educação na tecnologia e a aplicação prática de soluções em saúde. Um agradecimento especial a todos os desenvolvedores independentes que tornam seus conhecimentos públicos.

---

## ⚖️ Licença

Assim como as bases de imagens (HAM10000, BCN20000) e o repositório formador, este projeto encontra-se sob a licença **CC-BY-NC 4.0** (Não-Comercial). A replicação, estudo e uso acadêmico/pessoal são bem-vindos.
