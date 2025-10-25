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


# --- ETAPAS DO PIPELINE (COMO FUNÇÕES) ---

@st.cache_data(ttl=3600) # Faz cache do download e conversão por 1 hora
def download_e_converter_pdf(pdf_url):
    """Etapa 1 & 3: Baixa o PDF e converte para imagens."""
    try:
        pdf_response = requests.get(pdf_url)
        pdf_response.raise_for_status() 
        pdf_data = pdf_response.content
        images = convert_from_bytes(pdf_data)
        return images
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o PDF do link: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao converter o PDF (pdf2image/poppler): {e}")
        st.stop()

@st.cache_data(ttl=3600) # Também faz cache do OCR
def extrair_texto_das_imagens(images):
    """Etapa 4: Executa o OCR (Tesseract) em todas as imagens."""
    custom_config = r'--psm 6 -c load_system_dawg=0 -c load_freq_dawg=0'
    full_text = ""
    
    for i, page_image in enumerate(images):
        try:
            text = pytesseract.image_to_string(page_image, lang='por', config=custom_config)
            full_text += f"\n\n--- INÍCIO PÁGINA {i + 1} ---\n{text}"
        except Exception as e:
            st.warning(f"Erro no Tesseract (OCR) na página {i + 1}: {e}")
            continue
    return full_text

def extrair_dados_com_ia(full_text):
    """Etapa 5: Primeira chamada à IA para extrair o JSON."""
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
        return None # Retorna None em caso de falha

# --- (NOVO) PROMPT DE ANÁLISE RESUMIDA ---
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

    Use este formato exato em markdown:

    ### Nível de Risco Geral: [Crítico / Alerta / OK]

    **Ação Imediata (Crítico - Vermelho < 1.7mm):**
    * **Pneu [XX]:** Pior valor: [X.X]mm. [Ação recomendada do texto OCR, ex: "Substituir imediatamente"].
    * *(Liste todos os pneus críticos)*

    **Ação Recomendada (Alerta - Amarelo 1.7mm-3.0mm):**
    * **Pneu [XX]:** Pior valor: [X.X]mm. [Ação recomendada do texto OCR, ex: "Substituição recomendada"].
    * *(Liste todos os pneus em alerta)*

    **Pneus em Bom Estado (Verde > 3.0mm):**
    * **Pneu [XX]:** Pior valor: [X.X]mm.
    * *(Liste todos os pneus OK)*

    **Notas Adicionais:**
    * **Discos de Travão:** [Estado do texto OCR, ex: "Não verificado"].
    * **Alinhamento:** [Sugestão do texto OCR, ex: "Verificar alinhamento após substituição"].
    """
    
    try:
        response_analise = model_analise.generate_content(prompt_analise)
        return response_analise.text
    except Exception as e:
        st.error(f"Erro na IA (Análise): {e}")
        st.stop()

# --- (NOVA) FUNÇÃO DE UI PARA MÉTRICAS ---
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
    """(GOAL 1) Exibe a caixa de mensagem com os piores valores."""
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
        
        with col:
            st.metric(
                label=f"**{pneu}** (Diant. Esq.)" if pneu == "DE" else \
                      f"**{pneu}** (Diant. Dir.)" if pneu == "DD" else \
                      f"**{pneu}** (Tras. Esq.)" if pneu == "TE" else \
                      f"**{pneu}** (Tras. Dir.)",
                value=valor_display,
                delta=delta_label,
                delta_color=cor
            )

    # Adiciona o expander com os detalhes
    with st.expander("Ver todas as medições (Exterior / Centro / Interior)"):
        st.json(report_data)


# --- INTERFACE DO STREAMLIT (UI) ---

st.title("🤖 Analisador de Relatórios de Pneus (Autel TBE)")
st.write("Cole o link do relatório PDF gerado pelo QR Code para obter uma análise rápida para mecânicos.")

default_url = "https://gateway-prodeu.autel.com/api/pdf-report-manage/pdf-report/download/TB20M81009041758804644621"
pdf_url = st.text_input("URL do Relatório PDF:", value=default_url)

if st.button("Analisar Relatório", type="primary"):
    if not pdf_url:
        st.warning("Por favor, insira um URL.")
    else:
        report_data = None
        
        # Etapas 1-5 (Download, OCR, Extração IA)
        with st.spinner("A processar PDF e a extrair dados..."):
            images = download_e_converter_pdf(pdf_url)
            full_text = extrair_texto_das_imagens(images)
            report_data = extrair_dados_com_ia(full_text)
        
        if report_data:
            # (GOAL 1) Mostrar a caixa de métricas imediatamente
            mostrar_metricas_pneus(report_data)
            
            st.markdown("---") # Divisor

            # Etapa 6 (Análise Resumida IA)
            with st.spinner("A gerar relatório de ação resumido..."):
                final_report = gerar_relatorio_resumido_ia(full_text, json.dumps(report_data))
            
            # (GOAL 2) Mostrar o relatório resumido
            st.markdown(final_report)
        else:
            st.error("A extração de dados falhou. Não é possível gerar o relatório.")
