import streamlit as st

from main import build_system, run_query

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("GenAI Legal Assistant")

# -----------------------------------
# LOAD SYSTEM ONCE
# -----------------------------------

@st.cache_resource
def load_system():
    return build_system()

system = load_system()

# -----------------------------------
# USER INPUT
# -----------------------------------

query = st.text_input("Ask your legal question")

# -----------------------------------
# BUTTON
# -----------------------------------

if st.button("Submit"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing legal query..."):

            response = run_query(query, system)

        st.subheader("Answer")

        st.write(response)