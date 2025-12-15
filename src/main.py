import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from graph_rag import GraphRAG

# Extract only the raw content from LangChain / Ollama responses
def extract_content(output):
    try:
        # Most LangChain chains return FinalAnswer or AIMessage objects
        if hasattr(output, "content"):
            return str(output.content)
        # Some chains return dicts
        elif isinstance(output, dict) and "content" in output:
            return str(output["content"])
        # Fallback: check if string contains "Final Answer: content='…'"
        elif isinstance(output, str) and "Final Answer: content=" in output:
            import re
            match = re.search(r"Final Answer: content='(.*?)' additional_kwargs=", output, re.DOTALL)
            if match:
                return match.group(1)
        # Default: convert to string
        return str(output)
    except Exception:
        return str(output)


def main():
    # Streamlit setup
    st.markdown("""
        <style>
            .title {
                font-size: 48px;
                font-weight: bold;
                text-align: center;  /* Ensures text is centered inside the div */
                display: flex;
                justify-content: center;  /* Centers the content horizontally */
                align-items: center;  /* Centers the content vertically */
                height: 100px;  /* Adjust as needed for proper vertical alignment */
                width: 100%;
                white-space: nowrap;  /* Prevents the title from wrapping to the next line */
            }
        </style>
        <div class="title">DocNexus - RAG-based Document Analysis 🕸️</div>
    """, unsafe_allow_html=True)


    # Load PDF and process documents
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    uploaded_file = st.file_uploader("Upload your PDF here 👇:", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            documents = documents[:10]
            st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        # Container for chat history
        response_container = st.container()

        # Container for text box
        container = st.container()

        with container:
            if 'generated' in st.session_state and 'past' in st.session_state:
                with st.form(key='my_form', clear_on_submit=True):
                    query = st.text_input("Enter your query:", key='input')
                    submit_button = st.form_submit_button(label='Send')
                if submit_button and query:
                    # Only initialize and process once
                    if 'graph_rag' not in st.session_state:
                        graph_rag = GraphRAG()
                        graph_rag.process_documents(documents)
                        st.session_state['graph_rag'] = graph_rag

                    output = st.session_state['graph_rag'].query(query)
                    output_text = extract_content(output)
                    st.session_state.past.append(query)
                    st.session_state.generated.append(output_text)


            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

if __name__ == "__main__":
    main()
