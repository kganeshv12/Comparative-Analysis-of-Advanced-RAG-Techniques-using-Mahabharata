import os
import dotenv
from langchain import hub
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.load import dumps, loads
from langchain_community.llms import Cohere
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import CohereRerank, CohereRagRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


import openai
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore




dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def Chunk_high(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = (
        {
            "context":retriever | format_docs,
            "question" : RunnablePassthrough()
        }
        | prompt
        |llm
        | StrOutputParser()
    )

    ans = rag_chain.invoke(input_question)
    retrieved_docs = retriever.get_relevant_documents(input_question)
    context = format_docs(retrieved_docs)

    return ans, context

def Chunk_low(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_2",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = (
        {
            "context":retriever | format_docs,
            "question" : RunnablePassthrough()
        }
        | prompt
        |llm
        | StrOutputParser()
    )

    ans = rag_chain.invoke(input_question)
    retrieved_docs = retriever.get_relevant_documents(input_question)
    context = format_docs(retrieved_docs)

    return ans, context

def Rerank(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    compressor = CohereRerank(cohere_api_key= COHERE_API_KEY ,model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    def compressed_docs(question):
        intermediate = compression_retriever.get_relevant_documents(question)
        # print(intermediate)
        return intermediate
    
    rag_chain = (
        {
            "context": lambda x: format_docs(compressed_docs(x)),  # Compose functions properly
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(input_question)
    context = format_docs(compressed_docs(input_question))

    return answer, context

def Decomposition(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)


    generate_queries_decomposition  = (
        prompt_decomposition 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    questions = generate_queries_decomposition.invoke({"question":input_question})

    prompt_rag = hub.pull("rlm/rag-prompt")

    def retrieve_and_rag(question,prompt_rag,sub_question_generator_chain):
        """RAG on each sub-question"""
        
        # Use our decomposition / 
        sub_questions = sub_question_generator_chain.invoke({"question":question})
        
        # Initialize a list to hold RAG chain results
        rag_results = []

        rag_contexts = ""
        
        for sub_question in sub_questions:
            
            # Retrieve documents for each sub-question
            retrieved_docs = retriever.get_relevant_documents(sub_question)
            
            # Use retrieved documents and sub-question in RAG chain
            answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                    "question": sub_question})
            rag_results.append(answer)
            rag_contexts = rag_contexts + "\n\n" + "\n Sub Question : " + sub_question + "\n Retrieved Documents : " + format_docs(retrieved_docs) + "\n Answer : " + answer
        
        return rag_results,sub_questions, rag_contexts

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions, context_1 = retrieve_and_rag(input_question, prompt_rag, generate_queries_decomposition)

    def format_qa_pairs(questions, answers):
        """Format Q and A pairs"""
        
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    context_2 = format_qa_pairs(questions, answers)

    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    ans = final_rag_chain.invoke({"context": context_2, "question": input_question})

    context = context_1 + "\n\n" + context_2

    return ans, context

def Hyde(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    template = """Imagine you are an expert writing a detailed explanation on the topic: '{question}'
            Your response should be comprehensive and include all key points that would be found in the top search result."""
    prompt_hyde = ChatPromptTemplate.from_template(template)


    generate_docs_for_retrieval = (
        prompt_hyde | llm | StrOutputParser() 
    )   
    # generate_docs_for_retrieval.invoke({"question":question})

    retrieval_question = generate_docs_for_retrieval.invoke({"question":input_question})
    retireved_docs = retriever.get_relevant_documents(retrieval_question)
    context = format_docs(retireved_docs)


    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    ans = final_rag_chain.invoke({"context":context,"question":input_question})

    return ans, context

def MultiQueryRetrievers(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)   

    generate_queries = (
        prompt_perspectives 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    def get_unique_union(documents: list[list]):

        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))

        return [loads(doc) for doc in unique_docs]

    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":input_question})
    context = format_docs(docs)

    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    ans = final_rag_chain.invoke({"question":input_question, "context": context})


    return ans, context 

def RagFusion(model_name, input_question):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)


    generate_queries = (
        prompt_rag_fusion 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    ## Chain for extracting relevant documents

    retrieval_chain_rag_fusion = generate_queries | retriever.map()

    # retrieve documents
    results = retrieval_chain_rag_fusion.invoke({"question": input_question})

    print(len(results))

    lst=[]
    for ddxs in results:
        for ddx in ddxs:
            if ddx.page_content not in lst:
                lst.append(ddx.page_content)

    
    fused_scores = {}
    k=60
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            # print('\n')
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # final reranked result
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (prompt
        | llm
        | StrOutputParser()
    )

    def format_RF_docs(docs):
        return "\n\n".join(doc[0].page_content for doc in docs)
    
    context = format_RF_docs(reranked_results)

    ans = final_rag_chain.invoke({"context":context,"question":input_question})

    return ans, context

def SemanticRouting(model_name, input_question):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    Factual_template = """You are a very good in answering FACTUAL questions regarding Indian Epics. \
    You are great at answering these Factual questions pertaining to either Mahabharata or Ramayana in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Context:
    {context}

    Here is a question:
    {query}"""

    Interpretation_template = """You are a very good Analyser. You are great at answering Interpretation or Inference based questions from Indian Epics like Mahabharata and Ramayana. \
    You are so good because you are able to break down hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Context:
    {context}

    Here is a question:
    {query}"""

    Long_Ans_template = """You are a very good in provisinf In Detail, Long answers capturing all the aspects of the given question. You are great at answering Long Answer Questions from Indian Epics like Mahabharata and Ramayana. \
    You are so good because you are touch upon all important and relevant points, \
    answer the question in depth with all relevant details and facts. \

    Context:
    {context}

    Here is a question:
    {query}"""

    prompt_templates = [Factual_template, Interpretation_template, Long_Ans_template]
    prompt_embeddings = embeddings.embed_documents(prompt_templates)

    def prompt_router(input):
        # Embed question
        query_embedding = embeddings.embed_query(input["query"])
        # Compute similarity
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        most_similar = prompt_templates[similarity.argmax()]
        # Chosen prompt 
        if most_similar == Factual_template:
            print("Using Facts : ")  
        elif most_similar == Interpretation_template:
            print("Using Interpretations : ")
        else : print("Using Long Answers : ")
        return PromptTemplate.from_template(most_similar)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "query": RunnablePassthrough()
        }
        | RunnableLambda(prompt_router)
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(input_question)
    context = format_docs(retriever.get_relevant_documents(input_question))
  
    return answer, context

def SentenceWindowRetriever(model_name, input_question):

    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")

    llm = Groq(
        groq_api_key=GROQ_API_KEY,
        model=model_name,
        context_window=32000,
        max_tokens = 8000
    )

    def get_sentence_window_index(documents, index_dir, sentence_window_size=3):
        Node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_sentence",
        )

        Settings.llm = llm
        Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY,  model="text-embedding-3-small")
        Settings.node_parser = Node_parser

        if not os.path.exists(index_dir):
            db = chromadb.PersistentClient(path=index_dir)
            chroma_collection = db.get_or_create_collection("Sentence_Window_retriever")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=Settings.embed_model
            )
            
        else:
            print("using existing db")
            db2 = chromadb.PersistentClient(path=index_dir)
            chroma_collection = db2.get_or_create_collection("Sentence_Window_retriever")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=Settings.embed_model,
                node_parser=Settings.node_parser,
                show_progress=True
            )

        return index
    
    def get_sentence_window_engine(sentence_index):
    
        postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window",)
        rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base") 
        sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=6, node_postprocessors=[postprocessor, rerank])
        
        return sentence_window_engine
    
    index_dir = "../RAG_models/chroma_llamaindex_db_SWR"
    sw_index_2 = get_sentence_window_index("a", index_dir, sentence_window_size=3)
    sw_engine_2 = get_sentence_window_engine(sw_index_2)

    window_response_3 = sw_engine_2.query(input_question)
    answer = window_response_3.response

    def extract_context_from_nodes(window_response):

        combined_context = "\n\n".join([
            node.get_content() for node in window_response.source_nodes
        ])
        
        return combined_context
    
    context = extract_context_from_nodes(window_response_3)


    return answer, context

def StepBackPrompt(model_name, input_question):
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        max_tokens = 8000,
        temperature=0.1,
    )


    vector_store = Chroma(
    persist_directory="../RAG_models/chroma_langchain_db_1",
    embedding_function=embeddings
    )

    retriever  = vector_store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel's was born in what country?",
            "output": "what is Jan Sindel's personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at answering all kind of Questions, especially Interpretation based and logical answering pertaining to Mahabharata. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )

    generate_queries_step_back = prompt | llm | StrOutputParser()

    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    {normal_context}
    {step_back_context}

    Original Question: {question}
    Answer:"""

    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    def retrieve_contexts(question):
        """
        Retrieve contexts for both the original and step-back questions.
        
        Args:
            question (str): The original input question
        
        Returns:
            dict: A dictionary containing retrieved contexts
        """
        # Generate step-back query
        step_back_query = generate_queries_step_back.invoke({"question": question})
        
        # Retrieve contexts
        normal_context_docs = retriever.invoke(question)
        step_back_context_docs = retriever.invoke(step_back_query)
        
        # Format contexts
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        return {
            "normal_context_docs": normal_context_docs,
            "step_back_context_docs": step_back_context_docs,
            "normal_context": format_docs(normal_context_docs),
            "step_back_context": format_docs(step_back_context_docs)
        }

    # Modified chain to include context retrieval
    def rag_with_contexts(question):
        """
        Perform RAG with step-back query and retrieve contexts.
        
        Args:
            question (str): The input question
        
        Returns:
            dict: Response with contexts and answer
        """
        # Retrieve contexts
        contexts = retrieve_contexts(question)
        
        # Prepare input for response prompt
        response_input = {
            "normal_context": contexts['normal_context'],
            "step_back_context": contexts['step_back_context'],
            "question": question
        }
        
        # Generate response
        answer = (response_prompt | llm | StrOutputParser()).invoke(response_input)
        
        # Combine all information
        return {
            "original_question": question,
            "step_back_query": generate_queries_step_back.invoke({"question": question}),
            "normal_context_docs": contexts['normal_context_docs'],
            "step_back_context_docs": contexts['step_back_context_docs'],
            "normal_context": contexts['normal_context'],
            "step_back_context": contexts['step_back_context'],
            "answer": answer
        }

    # Usage
    result = rag_with_contexts(input_question)

    answer = result['answer']

    context = result['normal_context'] + result['step_back_context']

    return answer, context








