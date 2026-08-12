# 🔬 Melanoma-IA-Dermatoscópio

![Submissão Latinoware](https://img.shields.io/badge/Submiss%C3%A3o-Latinoware-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-CC--BY--NC_4.0-lightgrey?style=for-the-badge)

Um projeto de Inteligência Artificial focado na classificação binária de lesões cutâneas e detecção precoce de melanoma, desenvolvido com a poderosa arquitetura de Visão Computacional (Vision Transformer) **EVA-02**. 

Este repositório é parte de um projeto submetido ao evento **Latinoware**, visando demonstrar o enorme potencial de soluções Open Source de Deep Learning no auxílio à análise dermatológica, integrando software avançado com hardware acessível.

> **⚠️ AVISO MÉDICO IMPORTANTE:** Este sistema foi desenvolvido estritamente como uma **ferramenta de pesquisa** (Research Tool). **NÃO possui validação clínica** e jamais deve ser utilizado para substituir um diagnóstico profissional. Consulte sempre um médico dermatologista.

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
# Se o seu PC possui Placa de Vídeo NVIDIA (CUDA), utilize este comando:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Em seguida, instale as demais bibliotecas gráficas, modelos e dependências:
pip install timm pillow numpy huggingface_hub
```
*(Nota: Se o seu computador rodar em macOS, Linux/Raspberry Pi ou Windows sem placa de vídeo dedicada, você pode instalar o PyTorch tradicional através do comando `pip install torch torchvision`).*

---

## 🚀 Como Utilizar 

O projeto oferece diferentes abordagens de uso, dependendo da sua necessidade: desde análises estáticas via terminal até captura em tempo real usando hardware externo.

### 📸 Integração com Hardware: Raspberry Pi 4 com OV5647 (`auto_pipeline.py`)

Um dos grandes diferenciais deste projeto é a capacidade de realizar a captura e análise de forma automatizada através de dispositivos embarcados. O script `auto_pipeline.py` foi estruturado para interagir com a câmera do Raspberry Pi 4(ideal para módulos como Picamera2), recebendo o fluxo de imagens do dermatoscópio acoplado e enviando para o modelo classificar em poucos segundos.

**Rotina de Captura e Análise:**

Ao iniciar o script, o sistema executa um ciclo de monitoramento onde captura 1 frame da câmera a cada 2 segundos, durante um período total de 20 segundos. Cada imagem capturada é processada pela inteligência artificial, e os resultados da classificação são salvos localmente em um arquivo de texto (.txt) e enviados automaticamente para o Raspberry Pi via requisição HTTP logo após as 10 análises.

**Uso do pipeline:**
```bash
py auto_pipeline.py
```
*(Certifique-se de que o ip, usuário e senha do Raspberry Pi estão corretos).*

### 💻 Inferência Estática (`predict.py`)

Para testar imagens já salvas no seu computador dentro da raiz do projeto, o repositório conta com um script de inferência profissional completo, que suporta análise de imagem única.

**Uso - Imagem Única:**
```bash
python predict.py --checkpoint lesao.jpg --image lesao.jpg
```

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
