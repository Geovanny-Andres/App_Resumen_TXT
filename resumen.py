import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

# Función para cargar el modelo de lenguaje con la API Key de OpenAI
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Configuración de la página de Streamlit
st.set_page_config(page_title="Resumen de Texto en Español")
st.header("AI Resumen de Textos en Español")

# Instrucciones
col1, col2 = st.columns(2)

with col1:
    st.markdown("Esta app permite resumir textos largos usando inteligencia artificial. Ideal para estudiantes, investigadores y curiosos.")

with col2:
    st.write("Contacto: [AI Accelera](https://aiaccelera.com)")

# Ingreso de la API Key
st.markdown("## Ingresa tu clave de OpenAI")

def get_openai_api_key():
    input_text = st.text_input(label="Clave OpenAI", placeholder="Ej: sk-...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()

# Subida del archivo de texto
st.markdown("## Sube el archivo de texto que deseas resumir")

uploaded_file = st.file_uploader("Selecciona un archivo (.txt)", type="txt")

# Área para mostrar el resumen
st.markdown("### Aquí está tu resumen:")

if uploaded_file is not None:
    # Leer el archivo como string
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    if len(file_input.split(" ")) > 20000:
        st.write("Por favor sube un archivo más corto. El límite es 20.000 palabras.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Por favor ingresa tu clave de OpenAI. \
            Instrucciones [aquí](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

        # Dividir el texto en fragmentos para procesar
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=5000, 
            chunk_overlap=350
        )
        splitted_documents = text_splitter.create_documents([file_input])

        # Cargar el modelo con la API Key
        llm = load_LLM(openai_api_key=openai_api_key)

        # Crear prompt personalizado para resumir en español
        custom_prompt = PromptTemplate.from_template("Resume el siguiente texto en español:\n{text}")

        # Cargar la cadena de resumen con el prompt personalizado
        summarize_chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce",
            combine_prompt=custom_prompt
        )

        # Ejecutar el resumen
        summary_output = summarize_chain.run(splitted_documents)

        # Mostrar el resultado
        st.write(summary_output)
