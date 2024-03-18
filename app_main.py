import time
import datetime
from inference import qa_bot

import streamlit as st

@st.cache(allow_output_mutation=True)
def load_qa_bot():
       return qa_bot()

def ask_bot(query):
        start_time = time.time()
        response = qa_result({"query":query})
        print('---------> Retrieval time :',f"{str(datetime.timedelta(seconds=(time.time() - start_time)))} seconds.")
        return response

if __name__=='__main__':
    qa_result = load_qa_bot()  #-try moving this to line 7 out of main function, as every time query is asking it is loading model again

    st.title("Welcome to the QA bot")
    st.subheader("I can answer any questions related to COVID-19 Open Research")
    
    if "messages" not in st.session_state:
               st.session_state.messages = []

    for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if query := st.chat_input("Ask any questions realted to CORD19 dataset?"):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                       st.markdown(query)
                
                with st.chat_message("assistant"):                        
                        response = ask_bot(query)
                        # response = st.write_stream(stream)
                        st.write(response['result'])
                st.session_state.messages.append({"role": "assistant", "content": response})


