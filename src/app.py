import streamlit as st
import inspect
import RAG_Methods

def get_rag_methods():
    """
    Dynamically discover RAG methods in the RAG_Methods module.
    Only considers functions that take model_name and input_question as parameters.
    
    Returns:
        dict: A dictionary of method names to their corresponding functions
    """
    rag_methods = {}
    
    # Inspect all attributes in the RAG_Methods module
    for name, obj in inspect.getmembers(RAG_Methods):
        # Check if it's a function
        if inspect.isfunction(obj):
            # Get the function's signature
            signature = inspect.signature(obj)
            parameters = list(signature.parameters.keys())
            
            # Check if the function has the expected signature
            if (len(parameters) >= 2 and 
                parameters[0] == 'model_name' and 
                parameters[1] == 'input_question'):
                rag_methods[name] = obj
    
    return rag_methods

def initialize_chat_history():
    """Initialize or retrieve chat history from session state."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_chat_history():
    """Display the current chat history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    # Dynamically get RAG methods
    rag_methods = get_rag_methods()
    
    # Set page configuration
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
    
    # Title
    st.title("💬 Mahabharata ChatBot")
    
    # Initialize chat history
    initialize_chat_history()
    
    # Sidebar for configuration
    st.sidebar.header("Chatbot Settings")
    
    # LLM Selection
    selected_llm = st.sidebar.selectbox(
        "Select LLM Model", 
        [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile", 
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "gemma2-9b-it",

        ]
    )
    
    # RAG Method Selection Dropdown with dynamically discovered methods
    selected_rag_method = st.sidebar.selectbox(
        "Select RAG Method", 
        list(rag_methods.keys())
    )
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using RAG method
        try:
            # Dynamically call the selected RAG method
            selected_method = rag_methods[selected_rag_method]
            
            # Check method's return signature and handle accordingly
            sig = inspect.signature(selected_method)
            return_count = len(sig.parameters) - 2  # Subtract model_name and input_question
            
            # Call the method with appropriate parameters
            if return_count == 1:
                response = selected_method(selected_llm, prompt)
                context = None
            else:
                # Assumes methods return (answer, context)
                response, context = selected_method(selected_llm, prompt)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Optional: Expandable context section
            if context is not None:
                with st.expander("Retrieved Context"):
                    st.write(context)
        
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_message}
            )
    
    # Display chat history
    display_chat_history()

    # Sidebar information
    st.sidebar.markdown("### Current Configuration")
    st.sidebar.write(f"**LLM:** {selected_llm}")
    st.sidebar.write(f"**RAG Method:** {selected_rag_method}")
    
    # Optional method information
    if selected_rag_method in rag_methods:
        method_doc = rag_methods[selected_rag_method].__doc__
        if method_doc:
            st.sidebar.markdown("### Method Description")
            st.sidebar.write(method_doc)

if __name__ == "__main__":
    main()
