# Ficheiro: app.py

import streamlit as st
import requests
import pytesseract
import re
from pdf2image import convert_from_bytes
import sys
import json 
import google.generativeai as genai
import textwrap

# --- NOVAS BIBLIOTECAS PARA QR CODE ---
from PIL import Image
from pyzbar.pyzbar import decode

# --- Configuração da Página e API Key ---
st.set_page_config(page_title="Analisador de Relatório Autel", page_icon=" mechanic")

# Configurar a API Key do Streamlit Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Erro: A GOOGLE_API_KEY não foi encontrada. Por favor, adicione-a aos 'Secrets' da sua aplicação Streamlit.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao configurar a API Key: {e}")
    st.stop()


# --- (NOVA) FUNÇÃO PARA DESCODIFICAR QR CODE ---
def decode_qr_code(image_data):
    """Lê dados de uma imagem (de upload ou câmara) e descodifica o QR code."""
    try:
        # Abrir a imagem com o Pillow
        img = Image.open(image_data)
        
        # Descodificar QR codes
        decoded_objects = decode(img)
        
        if decoded_objects:
            # Assume que o primeiro QR code encontrado é o correto
            url = decoded_objects[0].data.decode('utf-8')
            return url
        else:
            return None
    except Exception as e:
        st.error(f"Erro ao processar a imagem do QR Code: {e}")
        return None

# --- ETAPAS DO PIPELINE (COMO FUNÇÕES) ---

@st.cache_data(ttl=3600)
def download_e_converter_pdf(pdf_url):
    try:
        pdf_response = requests.get(pdf_url)
        pdf_response.raise_for_status() 
        pdf_data = pdf_response.content
        images = convert_from_bytes(pdf_data)
        return images
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o PDF do link: {pdf_url}. Verifique o URL.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao converter o PDF (pdf2image/poppler): {e}")
        st.stop()

@st.cache_data(ttl=3600)
def extrair_texto_das_imagens(_images):
    custom_config = r'--psm 6 -c load_system_dawg=0 -c load_freq_dawg=0'
    full_text = ""
    for i, page_image in enumerate(_images):
        try:
            text = pytesseract.image_to_string(page_image, lang='por', config=custom_config)
            full_text += f"\n\n--- INÍCIO PÁGINA {i + 1} ---\n{text}"
        except Exception as e:
            st.warning(f"Erro no Tesseract (OCR) na página {i + 1}: {e}")
            continue
    return full_text

def extrair_dados_com_ia(full_text):
    model_extraca = genai.GenerativeModel('gemini-2.5-flash-lite')
    prompt_extracao = f"""
    Analise o seguinte texto, que foi extraído (via OCR) de um relatório de pneus.
    O texto pode conter erros de OCR (ex: '3.7' pode aparecer como '37', 'mm' pode aparecer como 'mni', 'TE' pode aparecer como 'SE').
    Sua tarefa é extrair as três medições de profundidade (em mm) para cada pneu: DE, DD, TE e TD.
    REGRAS IMPORTANTES:
    1. Os valores corretos são os 3 números que aparecem *embaixo* dos rótulos "DE", "DD", "TE", ou "TD" nas secções de inspeção detalhadas.
    2. Ignore os valores no sumário principal (ex: "8 mm", "3 mm", "1.6 mm").
    3. Se o OCR removeu o ponto decimal (ex: '37' em vez de '3.7'), re-insira o ponto. (ex: '37' -> '3.7', '18' -> '1.8', '14' -> '1.4').
    4. Se um pneu não for encontrado, use "N/A" para as 3 medições.
    Retorne APENAS um objeto JSON único (um dicionário), no seguinte formato exato:
    {{
      "DE": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}},
      "DD": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}},
      "TE": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}},
      "TD": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}}
    }}
    Texto para analisar:
    ---
    {full_text}
    ---
    """
    try:
        response_extracao = model_extraca.generate_content(prompt_extracao)
        json_text = response_extracao.text.strip().replace("```json", "").replace("```", "")
        report_data = json.loads(json_text) 
        return report_data
    except Exception as e:
        st.error(f"Erro na IA (Extração) ou ao processar o JSON: {e}")
        st.error(f"Texto recebido da IA: {response_extracao.text}")
        return None

# ==================================================================
# --- (PROMPT DE ANÁLISE CORRIGIDO E MAIS RIGOROSO) ---
# ==================================================================
def gerar_relatorio_resumido_ia(full_text, report_data_json):
    """Etapa 6: Segunda chamada à IA para gerar um relatório de AÇÃO."""
    model_analise = genai.GenerativeModel('gemini-2.5-flash-lite') 
    
    prompt_analise = f"""
    Aja como um mecânico-chefe a escrever notas rápidas para a sua equipa.
    Baseado no texto OCR e nos dados JSON extraídos, gere um **Relatório de Ação Resumido**.
    Seja direto, técnico e use bullet points.

    - Texto OCR (para contexto de sugestões):
    --- TEXTO OCR ---
    {full_text}
    --- FIM DO TEXTO OCR ---

    - Dados Extraídos (para valores):
    --- JSON DE DADOS ---
    {report_data_json}
    --- FIM DO JSON DE DADOS ---

    **REGRAS PARA EVITAR ALUCINAÇÕES:**
    1.  A sua tarefa é categorizar cada um dos quatro pneus (DE, DD, TE, TD) numa das três categorias de risco.
    2.  Para CADA pneu, calcule o seu **PIOR (menor) valor** a partir dos 3 valores no JSON de dados.
    3.  Use esse *único pior valor* para decidir a categoria:
        * **Crítico (Vermelho):** Pior valor <= 1.6mm
        * **Alerta (Amarelo):** Pior valor entre 1.7mm e 3.0mm
        * **Bom (Verde):** Pior valor > 3.0mm
    4.  **IMPORTANTE: Cada pneu (DE, DD, TE, TD) só pode aparecer UMA VEZ no relatório final, dentro da sua categoria correta.** Não liste as outras medições.
    5.  Para a "Ação recomendada", procure no TEXTO OCR o bloco de texto específico desse pneu (ex: o bloco debaixo de "DE" e "Inspeção visual"). Encontre a linha "Sugestões de reparação:" *dentro* desse bloco específico. Cite apenas a sugestão "1.".
    6.  Ignore o sumário "Estado do pneu:" no topo do texto OCR, pois ele pode ser confuso.

    **Use este formato exato em markdown:**

    ### Nível de Risco Geral: [Crítico / Alerta / OK]

    **Ação Imediata (Crítico - Vermelho <= 1.6mm):**
    * **Pneu [XX]:** Pior valor: [X.X]mm. (Sugestão: [Ação recomendada "1." do bloco OCR desse pneu]).
    * *(Liste APENAS os pneus que se encaixam aqui)*

    **Ação Recomendada (Alerta - Amarelo 1.7mm-3.0mm):**
    * **Pneu [XX]:** Pior valor: [X.X]mm. (Sugestão: [Ação recomendada "1." do bloco OCR desse pneu]).
    * *(Liste APENAS os pneus que se encaixam aqui)*

    **Pneus em Bom Estado (Verde > 3.0mm):**
    * **Pneu [XX]:** Pior valor: [X.X]mm. (Sugestão: [Ação recomendada "1." do bloco OCR desse pneu, ex: "Verificar pneus regularmente"]).
    * *(Liste APENAS os pneus que se encaixam aqui)*

    **Notas Adicionais:**
    * **Discos de Travão:** [Estado do texto OCR, ex: "Não verificado"].
    * **Alinhamento:** [Mencione a sugestão de "alinhamento" do texto OCR, se existir].
    """
    
    try:
        response_analise = model_analise.generate_content(prompt_analise)
        return response_analise.text
    except Exception as e:
        st.error(f"Erro na IA (Análise): {e}")
        st.stop()


def get_cor_e_risco(valor_mm):
    """Define a cor e o delta para a métrica com base no risco."""
    if valor_mm is None:
        return "normal", "N/A"
    
    if valor_mm <= 1.6:
        return "inverse", "CRÍTICO" # Vermelho
    elif valor_mm <= 3.0:
        return "normal", "Alerta" # Amarelo (laranja no st.metric)
    else:
        return "off", "Bom" # Verde

def mostrar_metricas_pneus(report_data):
    """Exibe a caixa de mensagem com os piores valores."""
    st.subheader("Balanço Rápido (Pior Medição)")
    
    col1, col2, col3, col4 = st.columns(4)
    pneus = {"DE": col1, "DD": col2, "TE": col3, "TD": col4}
    
    piores_valores = {}

    for pneu, col in pneus.items():
        data = report_data.get(pneu)
        pior_valor_mm = None
        
        if data:
            try:
                # Converte todas as medições em float, ignora "N/A"
                medicoes = [float(m) for m in data.values() if str(m).replace('.', '', 1).isdigit()]
                if medicoes:
                    pior_valor_mm = min(medicoes)
            except Exception:
                pass # Mantém pior_valor_mm como None
        
        piores_valores[pneu] = pior_valor_mm # Guarda para o expander
        
        # Define a cor e o texto de ajuda
        cor, delta_label = get_cor_e_risco(pior_valor_mm)
        valor_display = f"{pior_valor_mm} mm" if pior_valor_mm is not None else "N/A"
        
        label_map = {
            "DE": "**DE** (Diant. Esq.)",
            "DD": "**DD** (Diant. Dir.)",
            "TE": "**TE** (Tras. Esq.)",
            "TD": "**TD** (Tras. Dir.)"
        }
        
        with col:
            st.metric(
                label=label_map.get(pneu, f"**{pneu}**"),
                value=valor_display,
                delta=delta_label,
                delta_color=cor
            )

    # Adiciona o expander com os detalhes
    with st.expander("Ver todas as medições (Exterior / Centro / Interior)"):
        st.json(report_data)

# --- (NOVA) FUNÇÃO DE PIPELINE PRINCIPAL ---
def run_analysis_pipeline(pdf_url):
    """Executa todo o processo de análise num determinado URL."""
    report_data = None
    
    # Etapas 1-5 (Download, OCR, Extração IA)
    with st.spinner("A processar PDF e a extrair dados..."):
        images = download_e_converter_pdf(pdf_url)
        full_text = extrair_texto_das_imagens(images)
        report_data = extrair_dados_com_ia(full_text)
    
    if report_data:
        # Mostrar a caixa de métricas
        mostrar_metricas_pneus(report_data)
        st.markdown("---") # Divisor

        # Etapa 6 (Análise Resumida IA)
        with st.spinner("A gerar relatório de ação resumido..."):
            final_report = gerar_relatorio_resumido_ia(full_text, json.dumps(report_data))
        
        # Mostrar o relatório resumido
        st.markdown(final_report)
    else:
        st.error("A extração de dados falhou. Não é possível gerar o relatório.")


# --- INTERFACE DO STREAMLIT (UI) ---

st.title("🤖 Analisador de Relatórios de Pneus (Autel TBE)")
st.write("Forneça o relatório PDF usando uma das opções abaixo.")

# Link de exemplo para facilitar o teste
default_url = "https://gateway-prodeu.autel.com/api/pdf-report-manage/pdf-report/download/TB20M81009041758804644621"

tab1, tab2, tab3 = st.tabs(["Colar URL", "Upload QR Code", "Escanear QR Code"])

# --- Aba 1: Colar URL ---
with tab1:
    st.subheader("Opção 1: Colar o URL do Relatório")
    url_input = st.text_input("URL do Relatório PDF:", value=default_url)
    if st.button("Analisar por URL", type="primary"):
        if not url_input:
            st.warning("Por favor, insira um URL.")
        else:
            run_analysis_pipeline(url_input)

# --- Aba 2: Upload de Imagem QR Code ---
with tab2:
    st.subheader("Opção 2: Fazer Upload de uma Imagem do QR Code")
    qr_file = st.file_uploader("Carregue a foto do QR Code:", type=["png", "jpg", "jpeg"])
    
    if qr_file:
        # Tenta descodificar assim que o ficheiro é carregado
        with st.spinner("A ler o QR Code..."):
            pdf_url_from_qr = decode_qr_code(qr_file)
            
            if pdf_url_from_qr:
                st.success(f"QR Code lido com sucesso!")
                st.info(f"URL encontrado: `{pdf_url_from_qr}`")
                
                # Botão de análise para esta aba
                if st.button("Analisar a partir do QR Code (Upload)", type="primary"):
                    run_analysis_pipeline(pdf_url_from_qr)
            else:
                st.error("Não foi possível encontrar ou ler um QR Code na imagem carregada.")

# --- Aba 3: Escanear QR Code ---
with tab3:
    st.subheader("Opção 3: Escanear o QR Code com a Câmara")
    st.info("Permita o acesso à câmara e tire uma foto nítida do QR Code.")
    
    qr_cam_img = st.camera_input("Apontar a câmara para o QR Code")
    
    if qr_cam_img:
        # Tenta descodificar assim que a foto é tirada
        with st.spinner("A ler o QR Code da foto..."):
            pdf_url_from_cam = decode_qr_code(qr_cam_img)
            
            if pdf_url_from_cam:
                st.success(f"QR Code lido com sucesso!")
                st.info(f"URL encontrado: `{pdf_url_from_cam}`")
                
                # Botão de análise para esta aba
                if st.button("Analisar a partir do QR Code (Câmara)", type="primary"):
                    run_analysis_pipeline(pdf_url_from_cam)
            else:
                st.error("Não foi possível encontrar ou ler um QR Code na foto tirada.")
