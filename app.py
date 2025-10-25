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
st.set_page_config(page_title="Analisador de Relatório Autel", page_icon="📄")

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
    model_extraca = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt_extracao = f"""
    Analise o seguinte texto, que foi extraído (via OCR) de um relatório de pneus.
    O texto pode conter erros de OCR (ex: '3.7' pode aparecer como '37', 'mm' pode aparecer como 'mni', 'TE' pode aparecer como 'SE').
    Sua tarefa é extrair as três medições de profundidade (em mm) para cada pneu: DE, DD, TE e TD.
    REGRAS IMPORTANTES:
    1. Os valores corretos são os 3 números que aparecem *embaixo* dos rótulos "DE", "DD", "TE", ou "TD" nas secções de inspeção detalhadas.
    2. Ignore os valores no sumário principal (ex: "8 mm", "3 mm", "1.6 mm").
    3. Se o OCR removeu o ponto decimal (ex: '37' em vez de '3.7'), re-insira o ponto. O único valor que pode ser um inteiro é '14'. (ex: '37' -> '3.7', '18' -> '1.8', '14' -> '14').
    4. Se um pneu não for encontrado, use "N/A" para as 3 medições.
    Retorne APENAS um objeto JSON único (um dicionário), no seguinte formato exato:
    {{
      "DE": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}},
      "DD": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}},
      "TE": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "X.X"}},
      "TD": {{"medicao_1": "X.X", "medicao_2": "X.X", "medicao_3": "XX"}}
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
        st.stop()

def gerar_relatorio_com_ia(full_text, report_data):
    """Etapa 6: Segunda chamada à IA para gerar a análise."""
    model_analise = genai.GenerativeModel('gemini-1.5-pro-latest') 
    prompt_analise = f"""
    Aja como um especialista em segurança automóvel e mecânico de pneus.
    Eu tenho dois conjuntos de dados:
    
    1. O texto completo (com erros de OCR) de um relatório de pneus:
    --- TEXTO OCR ---
    {full_text}
    --- FIM DO TEXTO OCR ---

    2. Um JSON com os dados de medição extraídos desse texto:
    --- JSON DE DADOS ---
    {json.dumps(report_data)}
    --- FIM DO JSON DE DADOS ---

    Sua tarefa é gerar um relatório analítico detalhado em português, como se estivesse a explicar a um cliente.
    Use esta estrutura (use markdown para formatar):
    
    **Relatório Analítico de Pneus e Travões**

    **Sumário Executivo:**
    (Faça um resumo de 1-2 frases sobre o estado geral do veículo, com base na sugestão "Estado do pneu" que está no texto OCR.)

    **Análise Detalhada (Profundidade da Banda):**
    (Para cada pneu (DE, DD, TE, TD), liste a pior medição (o menor número) dos 3 valores. Explique o que isso significa. Use a legenda do relatório:
    - Verde (bom): > 3mm (ex: TE)
    - Amarelo (alerta): 1.7mm a 3mm (ex: DE, DD)
    - Vermelho (crítico): <= 1.6mm (ex: TD, se for o caso))

    **Recomendações e Próximos Passos:**
    (Liste as "Sugestões de reparação" encontradas no texto OCR para os pneus críticos (DD e TD) e os pneus em alerta (DE). Explique por que a "Distância de travagem" aumenta tanto com pneus gastos.)

    **Inspeção Adicional:**
    (Mencione o estado dos "Discos de travão" com base no texto OCR (ex: "Não verificado").)
    """
    
    try:
        response_analise = model_analise.generate_content(prompt_analise)
        return response_analise.text
    except Exception as e:
        st.error(f"Erro na IA (Análise): {e}")
        st.stop()


# --- INTERFACE DO STREAMLIT (UI) ---

st.title("🤖 Analisador de Relatórios de Pneus (Autel TBE)")
st.write("Cole o link do relatório PDF gerado pelo QR Code para obter uma análise completa.")

# Link de exemplo para facilitar o teste
default_url = "https://gateway-prodeu.autel.com/api/pdf-report-manage/pdf-report/download/TB20M81009041758804644621"
pdf_url = st.text_input("URL do Relatório PDF:", value=default_url)

if st.button("Analisar Relatório"):
    if not pdf_url:
        st.warning("Por favor, insira um URL.")
    else:
        # Estado 1: Download e OCR
        with st.status("Etapa 1/4: A baixar e processar o PDF...", expanded=True) as status:
            images = download_e_converter_pdf(pdf_url)
            st.write(f"PDF processado. {len(images)} página(s) encontradas.")
            
            status.update(label="Etapa 2/4: A extrair texto das imagens (OCR)...")
            full_text = extrair_texto_das_imagens(images)
            st.write("Texto extraído com sucesso.")
            status.update(state="complete", expanded=False)

        # Estado 2: Extração IA
        with st.status("Etapa 3/4: A extrair dados com a IA (Gemini)...") as status:
            report_data = extrair_dados_com_ia(full_text)
            st.write("Dados extraídos:")
            st.json(report_data) # Mostra o JSON extraído
            status.update(state="complete", expanded=False)

        # Estado 3: Análise IA
        with st.status("Etapa 4/4: A gerar relatório analítico (IA)...") as status:
            final_report = gerar_relatorio_com_ia(full_text, report_data)
            status.update(state="complete", expanded=True)
            
        st.success("Relatório Concluído!")
        st.markdown("---")
        
        # Exibe o relatório final
        st.markdown(final_report)
