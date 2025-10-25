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
        img = Image.open(image_data)
        decoded_objects = decode(img)
        if decoded_objects:
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
    """Etapa 5: Primeira chamada à IA para extrair o JSON de medições."""
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
# --- (NOVA) ETAPA 5.5: O MOTOR DE LÓGICA FEITO EM PYTHON ---
# ==================================================================
def analisar_dados_logicamente(report_data, full_text):
    """Pega no JSON de dados e no texto OCR e faz a análise lógica (0% IA)."""
    analysis_result = {}
    pneus = ["DE", "DD", "TE", "TD"]
    
    # --- Regras de Risco (Lógica Python) ---
    def get_categoria(valor_mm):
        if valor_mm is None:
            return "N/A"
        if valor_mm <= 1.6:
            return "Crítico"
        elif valor_mm <= 3.0:
            return "Alerta"
        else:
            return "Bom"

    # --- RegEx para encontrar a sugestão correta (Lógica Python) ---
    def get_sugestao(pneu, full_text):
        # Procura o bloco de texto específico do pneu 
        # (ex: "DE... Desgaste disc travão: ... Sugestões de reparação: 1. ... 2. ...")
        padrao = rf"{pneu}\s*.*?(?:Sugestões de reparação:|Sugestões:)\s*1\.\s*([^\n2\.]*)"
        match = re.search(padrao, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            sugestao = match.group(1).strip().replace('\n', ' ').replace('Sugestões', '')
            return sugestao
        return "Nenhuma sugestão específica encontrada."

    # --- Processa cada pneu ---
    categorias_encontradas = set()
    for pneu in pneus:
        data = report_data.get(pneu)
        pior_valor_mm = None
        medicoes_str = ["N/A", "N/A", "N/A"]

        if data:
            try:
                medicoes_str = [data.get('medicao_1', 'N/A'), data.get('medicao_2', 'N/A'), data.get('medicao_3', 'N/A')]
                medicoes_float = [float(m) for m in medicoes_str if str(m).replace('.', '', 1).isdigit()]
                if medicoes_float:
                    pior_valor_mm = min(medicoes_float)
            except Exception:
                pass
        
        categoria_pneu = get_categoria(pior_valor_mm)
        categorias_encontradas.add(categoria_pneu)
        
        analysis_result[pneu] = {
            "pior_valor": pior_valor_mm,
            "categoria": categoria_pneu,
            "sugestao": get_sugestao(pneu, full_text)
        }
    
    # --- Determina Risco Geral (Lógica Python) ---
    if "Crítico" in categorias_encontradas:
        risco_geral = "Crítico"
    elif "Alerta" in categorias_encontradas:
        risco_geral = "Alerta"
    elif "Bom" in categorias_encontradas:
        risco_geral = "OK"
    else:
        risco_geral = "Indeterminado"
        
    analysis_result["risco_geral"] = risco_geral
    
    # --- Processa Informações Adicionais (Lógica Python) ---
    discos_match = re.search(r"Desgaste disc travão:.*?(Não verificado)", full_text, re.IGNORECASE)
    alinhamento_match = re.search(r"(parâmetros de alinhamento das quatro rodas)", full_text, re.IGNORECASE)

    analysis_result["info_adicional"] = {
        "discos_travao": discos_match.group(1).strip() if discos_match else "Não mencionado",
        "alinhamento": "Recomenda-se verificar o alinhamento" if alinhamento_match else "Não mencionado"
    }
    
    return analysis_result


# ==================================================================
# --- ETAPA 6: A IA APENAS FORMATA O RELATÓRIO (SEM LÓGICA) ---
# ==================================================================
def gerar_relatorio_formatado_ia(analysis_json):
    """Etapa 6: A IA apenas FORMATA o JSON pré-analisado."""
    model_analise = genai.GenerativeModel('gemini-2.5-flash-lite') # Flash é suficiente para formatação
    
    # Este prompt é agora muito mais simples. A IA não pode errar.
    prompt_analise = f"""
    Aja como um formatador de relatórios. Você recebeu um objeto JSON que JÁ CONTÉM toda a lógica e análise de um relatório de pneus.
    Sua ÚNICA tarefa é formatar este JSON em um "Relatório de Ação Resumido" em markdown, em português.
    NÃO calcule, NÃO deduza, NÃO adicione informações que não estejam no JSON. Apenas formate o que foi dado.

    - JSON de Análise (Fonte da Verdade):
    --- JSON DE DADOS ---
    {analysis_json}
    --- FIM DO JSON DE DADOS ---

    **Instruções de Formatação:**
    1.  Use o "risco_geral" do JSON para o título.
    2.  Para cada categoria ("Crítico", "Alerta", "Bom"), liste os pneus que o JSON marcou com essa categoria.
    3.  Para cada pneu, liste seu "pior_valor" e a "sugestao" exata fornecida no JSON.
    4.  Se uma categoria não tiver pneus, escreva: "Nenhum pneu nesta categoria."
    5.  Nas "Notas Adicionais", liste as "info_adicional" do JSON.

    **Use este formato exato em markdown:**

    ### Nível de Risco Geral: [risco_geral do JSON]

    **Ação Imediata (Crítico - Vermelho <= 1.6mm):**
    * **Pneu [XX]:** Pior valor: [pior_valor]mm. (Sugestão: [sugestao]).
    * *(Liste aqui APENAS os pneus com categoria "Crítico")*

    **Ação Recomendada (Alerta - Amarelo 1.7mm-3.0mm):**
    * **Pneu [XX]:** Pior valor: [pior_valor]mm. (Sugestão: [sugestao]).
    * *(Liste aqui APENAS os pneus com categoria "Alerta")*

    **Pneus em Bom Estado (Verde > 3.0mm):**
    * **Pneu [XX]:** Pior valor: [pior_valor]mm. (Sugestão: [sugestao]).
    * *(Liste aqui APENAS os pneus com categoria "Bom")*

    **Notas Adicionais:**
    * **Discos de Travão:** [info_adicional.discos_travao do JSON]
    * **Alinhamento:** [info_adicional.alinhamento do JSON]
    """
    
    try:
        response_analise = model_analise.generate_content(prompt_analise)
        return response_analise.text
    except Exception as e:
        st.error(f"Erro na IA (Formatação): {e}")
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
                medicoes = [float(m) for m in data.values() if str(m).replace('.', '', 1).isdigit()]
                if medicoes:
                    pior_valor_mm = min(medicoes)
            except Exception:
                pass
        
        piores_valores[pneu] = pior_valor_mm 
        
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
        report_data_numeros = extrair_dados_com_ia(full_text) # JSON só com números
    
    if report_data_numeros:
        # (GOAL 1) Mostrar a caixa de métricas
        mostrar_metricas_pneus(report_data_numeros)
        st.markdown("---") # Divisor

        # (NOVA ETAPA 5.5) - Python faz a lógica
        with st.spinner("A analisar dados e sugestões..."):
            pre_analysis_json = analisar_dados_logicamente(report_data_numeros, full_text)
        
        # Etapa 6 (Análise Resumida IA)
        with st.spinner("A gerar relatório de ação formatado..."):
            final_report = gerar_relatorio_formatado_ia(json.dumps(pre_analysis_json))
        
        st.markdown(final_report)
    else:
        st.error("A extração de dados falhou. Não é possível gerar o relatório.")


# --- INTERFACE DO STREAMLIT (UI) ---

st.title("🤖 Analisador de Relatórios de Pneus (Autel TBE)")
st.write("Forneça o relatório PDF usando uma das opções abaixo.")

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
        with st.spinner("A ler o QR Code..."):
            pdf_url_from_qr = decode_qr_code(qr_file)
            
            if pdf_url_from_qr:
                st.success(f"QR Code lido com sucesso!")
                st.info(f"URL encontrado: `{pdf_url_from_qr}`")
                
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
        with st.spinner("A ler o QR Code da foto..."):
            pdf_url_from_cam = decode_qr_code(qr_cam_img)
            
            if pdf_url_from_cam:
                st.success(f"QR Code lido com sucesso!")
                st.info(f"URL encontrado: `{pdf_url_from_cam}`")
                
                if st.button("Analisar a partir do QR Code (Câmara)", type="primary"):
                    run_analysis_pipeline(pdf_url_from_cam)
            else:
                st.error("Não foi possível encontrar ou ler um QR Code na foto tirada.")
