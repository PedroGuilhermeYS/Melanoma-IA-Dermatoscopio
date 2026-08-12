import time
import io
import os
import requests
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from huggingface_hub import hf_hub_download
import paramiko

# ================= CONFIGURAÇÕES =================
RASP_IP = "192.168.100.131"                     # IP do seu Raspberry Pi 4
RASP_STREAM_URL = f"http://{RASP_IP}:5000/snapshot" # Rota snapshot no Rasp4
RASP_USER = "pi"                        # Usuário do Rasp4
RASP_PASSWORD = "pi"                    # Senha do Rasp4
RASP_DEST_DIR = "/home/pi/Downloads"    # Pasta no Rasp4

INTERVALO_SEGUNDOS = 2
FRAMES_POR_LOTE = 10
THRESHOLD = 0.430                               # Limiar recomendado (>= 0.430 -> MALIGNO)
MODEL_NAME = "eva02_small_patch14_336.mim_in22k_ft_in1k"
# =================================================

# Classe da arquitetura do modelo idêntica ao seu predict.py antigo
class ISICModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, drop_path_rate=0.1)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        return self.model(x)

def carregar_modelo_e_transform():
    """Baixa os pesos do Hugging Face e carrega o modelo na memória GPU/CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Carregando modelo EVA-02 no dispositivo: {device}...")
    
    # 1. Baixa os pesos do Hugging Face automaticamente
    ckpt_path = hf_hub_download(
        repo_id="fawo/eva02-small-melanoma-classifier", 
        filename="model_0001.pt"
    )
    
    # 2. Instancia o modelo e carrega o state_dict
    model = ISICModel(MODEL_NAME)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # 3. Cria a transformação exatamente como o modelo exige
    transform = create_transform(**resolve_data_config({}, model=model.model), is_training=False)

    return model, transform, device

def analisar_imagem(model, transform, device, img_pil):
    """Executa a inferência na imagem PIL direto da memória RAM."""
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor)).item()

    label = "MALIGNO" if prob >= THRESHOLD else "Benigno"
    return f"Diagnóstico: {label} (Probabilidade: {prob:.4f})"

def enviar_txt_para_rasp(caminho_local_txt, nome_arquivo_destino):
    """Envia o arquivo .txt via SFTP para o Raspberry Pi 4."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(RASP_IP, username=RASP_USER, password=RASP_PASSWORD)
        
        sftp = ssh.open_sftp()
        caminho_remoto = os.path.join(RASP_DEST_DIR, nome_arquivo_destino)
        sftp.put(caminho_local_txt, caminho_remoto)
        sftp.close()
        ssh.close()
        print(f"✅ Arquivo {nome_arquivo_destino} enviado com sucesso para o Rasp4!")
    except Exception as e:
        print(f"❌ Erro ao enviar para o Rasp4: {e}")

def loop_processamento():
    model, transform, device = carregar_modelo_e_transform()
    resultados_lote = []
    contador_frames = 0
    numero_lote = 1

    print("🚀 Automação iniciada! Pressione CTRL+C para parar.")

    while True:
        try:
            inicio = time.time()
            
            # 1. Captura a imagem atual do Rasp4 via HTTP (sem salvar no HD)
            response = requests.get(RASP_STREAM_URL, timeout=5)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                
                # 2. Executa a inferência na imagem em memória
                resultado = analisar_imagem(model, transform, device, image)
                
                contador_frames += 1
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                linha_log = f"Frame {contador_frames} [{timestamp}]: {resultado}\n"
                resultados_lote.append(linha_log)
                
                print(f"[{contador_frames}/{FRAMES_POR_LOTE}] Processado -> {resultado}")

                # 3. A cada 10 frames (20s), salva e envia o .txt
                if len(resultados_lote) >= FRAMES_POR_LOTE:
                    nome_txt = f"analise_lote_{numero_lote}.txt"
                    caminho_txt_local = os.path.join(".", nome_txt)
                    
                    with open(caminho_txt_local, "w", encoding="utf-8") as f:
                        f.writelines(resultados_lote)
                    
                    enviar_txt_para_rasp(caminho_txt_local, nome_txt)
                    
                    resultados_lote = []
                    numero_lote += 1

            tempo_decorrido = time.time() - inicio
            tempo_espera = max(0, INTERVALO_SEGUNDOS - tempo_decorrido)
            time.sleep(tempo_espera)

        except KeyboardInterrupt:
            print("\nEncerrando automação...")
            break
        except Exception as e:
            print(f"Erro no loop: {e}")
            time.sleep(2)

if __name__ == "__main__":
    loop_processamento()