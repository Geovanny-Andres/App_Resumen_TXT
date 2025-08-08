import streamlit as st # Carga Streamlit para crear la app web
from langchain import PromptTemplate # Permite crear plantillas de prompts
from langchain_openai import OpenAI # Integra modelos OpenAI con LangChain
from langchain.chains.summarize import load_summarize_chain # Carga cadena de resumen
from langchain.text_splitter import RecursiveCharacterTextSplitter # Divide texto largo en partes
import pandas as pd # Librería para manejo de datos (aunque no se usa aquí)
from io import StringIO # Para convertir archivos cargados en texto

# Función para cargar el modelo LLM con la API Key del usuario
#LLM and key loading function
def load_LLM(openai_api_key):
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key) # Modelo sin creatividad (respuestas precisas)
    return llm

# Configura el título de la página
#Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer") # Título principal de la app


# Muestra instrucciones y enlace de contacto en dos columnas
#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")


# Entrada para la API key de OpenAI
#Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")

# Función que genera un campo de entrada para escribir la API Key
def get_openai_api_key():
    # Campo de texto que solicita la clave API, ocultando los caracteres por seguridad
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text # Retorna el valor ingresado por el usuario

openai_api_key = get_openai_api_key() # Guarda la API Key del usuario


# Carga el archivo de texto a resumir
# Input
st.markdown("## Upload the text file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type="txt")

# Muestra título del resumen  
# Output
st.markdown("### Here is your Summary:")

# Solo se ejecuta si el usuario subió un archivo
if uploaded_file is not None:
    # Lee el archivo como bytes
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # Convierte el contenido del archivo a texto
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read() # Lee el contenido del archivo como string
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    file_input = string_data # Almacena el texto leído
    
    # Verifica que el texto no sea demasiado largo
    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20000 words.")
        st.stop()

    # Verifica que se haya ingresado la API Key
    if file_input:
        if not openai_api_key:
            st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

    # Divide el texto largo en fragmentos más pequeños
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], # Divide en saltos de línea
        chunk_size=5000, # Tamaño de cada fragmento
        chunk_overlap=350 # Cuánto se superpone entre fragmentos
        )
    # Aplica el divisor de texto al archivo cargado
    splitted_documents = text_splitter.create_documents([file_input])

     # Carga el modelo LLM
    llm = load_LLM(openai_api_key=openai_api_key)

   # Carga la cadena de resumen con estrategia map_reduce
    summarize_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce"
        )
    # Ejecuta la cadena para obtener el resumen del texto
    summary_output = summarize_chain.run(splitted_documents)
    # Muestra el resumen generado en la app
    st.write(summary_output)