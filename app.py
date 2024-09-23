import pandas as pd
import google.generativeai as genai
import streamlit as st
import time
from sqlalchemy import create_engine
from sqlalchemy import text
import os

GOOGLE_API = st.secrets["OPENAI_API_KEY"]
type = st.secrets["TYPE"]
id = st.secrets["ID"]

st.set_page_config(layout="wide",initial_sidebar_state="expanded")

def create_prompt_1(data):
  """
  This Function returns a prompt that informs GPT we want to generate SQL code.
  """
  prompt = f''' ### sqlite SQL tables, with their properties:

#
# data({format(",".join(str(x) for x in data.columns))})
#
'''
  return prompt

def combine_prompt_1(df,prompt):
    definition = create_prompt_1(df)
    query = f'### A query to answer: {prompt}\nSELECT'
    return definition+query

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)


genai.configure(api_key=os.environ["GOOGLE_API"])
# Choose a model that's appropriate for your use case.
model = genai.GenerativeModel('gemini-pro')


with st.sidebar:
    st.subheader("Upload Your Dataset in CSV",divider='rainbow')
    file = st.file_uploader("")
    if file:
        st.success('File Upload Successfully')
        df = pd.read_csv(file, index_col=None)
        temp_db = create_engine('sqlite:///:memory:',echo =True)
        data = df.to_sql('data',con=temp_db)

c1,c2 = st.columns([4,1])
with c1:
    st.header("NL-SQL",help="Natural Language to SQL code")

with c2:
    button = st.button("Delete session state")

if file:
    with st.expander("Dataset 1"):
        st.dataframe(df)
        st.text(f"Shape: {df.shape}")

prompt = st.chat_input("Enter the prompt for code generation")

if button:
    st.session_state.messages = []

else:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for messages in st.session_state.messages:
        with st.chat_message(messages["role"]):
            st.markdown(messages["content"])

    if prompt:
        step = combine_prompt_1(df,prompt)
        response = model.generate_content(step, stream=False)
        txt = response.text
        txt=txt[7:-4]
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role":"user","content": prompt})

        with st.chat_message("assistant"):
            st.write_stream(stream_data(txt))
            with temp_db.connect() as conn:
                result = conn.execute(text(txt))
                row = result.fetchall()
                with st.expander("Transformed Dataframe"):
                    row_df = pd.DataFrame(row)
                    st.dataframe(row_df)
                    st.text(f"Shape: {row_df.shape}")

        st.session_state.messages.append({"role":"assistant","content": txt})

        if 'sentences' not in st.session_state:
            st.session_state.sentences = []

        st.session_state.sentences.append(f" #### # {prompt} \n")
        st.session_state.sentences.append(f"{txt} \n\n")
        
        with st.sidebar:
            with st.popover("Code",use_container_width=True):
                for sentence in st.session_state.sentences:
                    st.write(sentence)
