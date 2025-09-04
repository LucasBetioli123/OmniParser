# ==============================================================================
# 1. IMPORTAÇÕES
# ==============================================================================
import time
import importlib
from PIL import Image
import torch
import numpy as np
import base64
import matplotlib.pyplot as plt
import io

# Importa as funções do projeto de forma limpa
from util import utils

# Opcional: Recarrega o módulo 'utils' se você estiver fazendo alterações nele
# importlib.reload(utils)

print("Bibliotecas importadas com sucesso.")

# ==============================================================================
# 2. CARREGAMENTO DOS MODELOS DE IA
# ==============================================================================
print("--- Carregando Modelos de IA (isso pode levar alguns minutos) ---")

# Define o dispositivo para usar a GPU, se disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Caminhos para os modelos (verifique se estão corretos)
YOLO_MODEL_PATH = 'weights/icon_detect/model.pt'
CAPTION_MODEL_PATH = "weights/icon_caption_florence"

# Carrega o modelo de detecção de objetos (YOLO-World)
som_model = utils.get_yolo_model(YOLO_MODEL_PATH)
som_model.to(device)
print(f"-> Modelo YOLO carregado em '{device}'.")

# Carrega o modelo de captioning de ícones (Florence-2)
caption_model_processor = utils.get_caption_model_processor(
    model_name="florence2",
    model_name_or_path=CAPTION_MODEL_PATH,
    device=device
)
print("-> Modelo Florence-2 carregado.")
print("--- Modelos prontos. ---")


# ==============================================================================
# 3. CONFIGURAÇÃO DA ANÁLISE (AJUSTADAS PARA MÁXIMA SENSIBILIDADE)
# ==============================================================================

# Defina o caminho da imagem que você quer analisar
image_path = r'c:\Users\lucas.betioli\Downloads\ondemand_layouts.png'

# Carrega a imagem
image = Image.open(image_path)
print(f"\nAnalisando a imagem: {image_path}, Tamanho: {image.size}")

# Configurações para a análise e para desenhar as caixas na imagem final
BOX_TRESHOLD = 0.00001  # Limiar de confiança muito baixo para detectar o máximo de objetos
box_overlay_ratio = max(image.size) / 3200
draw_bbox_config = {
    'text_scale': 0.2 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}

# ==============================================================================
# 4. EXECUÇÃO DO PIPELINE
# ==============================================================================
start = time.time()

# Etapa 1: OCR
print("\n[ETAPA 1/2] Executando OCR para extrair textos...")
ocr_bbox_rslt, is_goal_filtered = utils.check_ocr_box(
    image_path,
    display_img=False,
    output_bb_format='xyxy',
    goal_filtering=None,
    easyocr_args={'paragraph': False, 'text_threshold': 0.9},
    use_paddleocr=True
)
text, ocr_bbox = ocr_bbox_rslt
cur_time_ocr = time.time()
print(f"-> OCR finalizado em {cur_time_ocr - start:.2f} segundos.")

# Etapa 2: Análise Visual (SOM)
print("[ETAPA 2/2] Executando análise visual (SOM) e gerando descrições...")
dino_labled_img_base64, label_coordinates, parsed_content_list = utils.get_som_labeled_img(
    image_path,
    som_model,
    BOX_TRESHOLD=BOX_TRESHOLD,
    output_coord_in_ratio=True,
    ocr_bbox=ocr_bbox,
    draw_bbox_config=draw_bbox_config,
    caption_model_processor=caption_model_processor,
    ocr_text=text,
    use_local_semantics=True,
    iou_threshold=0.01,
    scale_img=False,  # Usar resolução total para máximo detalhe
    batch_size=256
)
cur_time_caption = time.time()
print(f"-> Análise visual finalizada em {cur_time_caption - cur_time_ocr:.2f} segundos.")

print(f"\nTempo total de processamento: {cur_time_caption - start:.2f} segundos.")

# ==============================================================================
# 5. EXIBIÇÃO DOS RESULTADOS
# ==============================================================================
print(f"\n--- Total de {len(parsed_content_list)} elementos detectados ---")
for item in parsed_content_list:
    print(item)

# Plota a imagem final com as anotações
try:
    image_result = Image.open(io.BytesIO(base64.b64decode(dino_labled_img_base64)))
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image_result)
    plt.show()

    # Salva a imagem final com as anotações
    output_image_path = "resultado_analise.png"
    image_result.save(output_image_path)
    print(f"\nImagem com as anotações foi salva em: {output_image_path}")

except Exception as e:
    print(f"Não foi possível exibir ou salvar a imagem. Erro: {e}")