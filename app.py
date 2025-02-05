# ANALÍTICA BOUTIQUE, SC (https://www.visoresanalitica.com.mx/)
# DEMO
# 
# PROBLEM: 

# What We Cover Today:
# 1. What is the SQL Database Agent App?
# 2. Expose you to my new AI Data Science Team (Army of Agents)
# 3. How to Build the App: SQL Database Agent App


# streamlit run app.py

# Imports
# pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI

import streamlit as st
import sqlalchemy as sql
import pandas as pd
import asyncio
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team.agents import SQLDatabaseAgent

# * APP INPUTS ----
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Configuración de la app en Streamlit
TITLE = "Agente que analiza una Base de Datos SQL"
st.set_page_config(page_title=TITLE, page_icon="📊")
st.title(TITLE)

st.markdown("""
Bienvenido al Agente de Base de Datos SQL. Este agente de IA está diseñado para ayudarte a consultar tu base de datos SQL y devolver data frames que puedes inspeccionar interactivamente y descargar.

El dataset de ejemplo es un listado de propiedades en Miami, FL y los counties adyacentes.

La fuente de los datos es Zillow.
""")

# Preguntas de ejemplo
with st.expander("Preguntas Ejemplo", expanded=False):
    st.write("""
    - ¿Cuál es el nombre de las columnas en la tabla de la base de datos? 
    - ¿Cuál es el precio promedio de una propiedad con 3 habitaciones? 
    - ¿Cuáles son las 10 propiedades más rentables? 
    - Agrupa las propiedades por tipo de propiedad.
    """)

# Configuración de la base de datos
DB_OPTIONS = {
    "Zillow-Miami Database": "sqlite:///data/Resultado_SEL.db",
}
db_option = st.sidebar.selectbox("Selecciona una base de datos", list(DB_OPTIONS.keys()))
st.session_state["PATH_DB"] = DB_OPTIONS.get(db_option)

# Conexión a la base de datos
sql_engine = sql.create_engine(st.session_state["PATH_DB"])
conn = sql_engine.connect()

# Modelos de OpenAI
MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']
model_option = st.sidebar.selectbox("Selecciona un modelo de OpenAI", MODEL_LIST, index=0)

# Inicializar el modelo de OpenAI
OPENAI_LLM = ChatOpenAI(model=model_option, api_key=openai_api_key)
llm = OPENAI_LLM

# Configurar la memoria del chat
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("¿En qué puedo ayudarte?")

# Almacenar DataFrames en session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Función para mostrar el historial de chat
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg.content)

# Renderizar mensajes previos
display_chat_history()

# Crear el Agente SQL
sql_db_agent = SQLDatabaseAgent(
    model=llm,
    connection=conn,
    n_samples=1,
    log=False,
    bypass_recommended_steps=True,
)

# Manejo de preguntas async
async def handle_question(question):
    await sql_db_agent.ainvoke_agent(user_instructions=question)
    return sql_db_agent

# Entrada de usuario en el chat
if st.session_state["PATH_DB"] and (question := st.chat_input("Escribe tu pregunta aquí:", key="query_input")):
    with st.spinner("Pensando..."):

        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        error_occured = False
        try:
            result = asyncio.run(handle_question(question))
        except Exception as e:
            error_occured = True
            response_text = f"""
            Lo siento. Estoy teniendo dificultades para responder esa pregunta. 
            
            Error: {e}
            """
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)
            st.error(f"Error: {e}")

        # Mostrar resultados si no hubo error
        if not error_occured:
            sql_query = result.get_sql_query_code()
            response_df = result.get_data_sql()

            if sql_query:
                response_1 = f"### Resultados SQL:\n\nConsulta SQL:\n\n```sql\n{sql_query}\n```\n\nResultado:"

                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(response_df)

                msgs.add_ai_message(response_1)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_1)
                st.dataframe(response_df)