import os
import streamlit as st
import torch
import pickle
import glob
import time
import uuid
from typing import List, Generator
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from elasticsearch import Elasticsearch
import numpy as np
import traceback
from langchain_community.chat_models import ChatTongyi
import html

# 重要：必须在任何 Streamlit 命令之前设置页面配置
st.set_page_config(
    page_title="科普知识问答助手",
    layout="centered",
    page_icon="🔬",
    initial_sidebar_state="collapsed"
)

# 解决streamlit的冲突问题
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# ======== 提示词统一管理 ========
PROMPTS = {
    "is_scientific_query": [
        SystemMessage(content="请判断以下问题是否属于'科普相关内容'（如科学、技术、演讲等），请务必只回答'是'或'否'："),
        HumanMessage(content="{question}")
    ],
    "rag_answer": [
        SystemMessage(content="你是理性且友好的科学问答助手。请根据以下科普内容，做个全面的总结和思考，回答用户问题。如果找不到答案，请回复'抱歉哦，我的知识库里没有找到相关的内容！'。"),
        HumanMessage(content="上下文：\n{context}\n\n问题：\n{question}")
    ],
    "simple_answer": [
        SystemMessage(content="你是一个活泼可爱又理性的科普问答小助手，请简洁、友好、礼貌地回答以下问题："),
        HumanMessage(content="{question}")
    ],
    "multi_turn": [
        SystemMessage(content="你是理性且友好的科学问答助手。请根据对话历史和以下科普内容回答问题：\n{context}"),
        HumanMessage(content="当前对话历史：\n{history}\n\n问题：\n{question}")
    ],
}

# 配置路径和环境变量
DATA_DIR = os.path.join("data_pdf", "merged_pdfs")
VECTOR_STORE_PATH = os.path.join("vector_store", "vector_store.pkl")
ES_INDEX_NAME = "rag_docs"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 设置成自己的密钥
os.environ["TONGYI_API_KEY"] = "sk-506a5c243ac3445caea6be389b0025fb"  

# 稳定DOM更新的消息管理器
class ChatMessageManager:
    def __init__(self):
        self.messages = {}
        
    def add_message(self, role, content, placeholder=None):
        """添加新消息，返回消息ID和占位符"""
        message_id = str(uuid.uuid4())
        self.messages[message_id] = {
            "role": role,
            "content": content,
            "placeholder": placeholder or st.empty()
        }
        return message_id
        
    def update_message(self, message_id, new_content):
        """安全更新消息内容"""
        if message_id in self.messages:
            try:
                # 使用HTML转义确保内容安全
                safe_content = html.escape(new_content).replace("\n", "<br>")
                
                # 构建HTML结构
                css_class = "user-message" if self.messages[message_id]["role"] == "user" else "assistant-message"
                html_content = f"<div class='{css_class}'>{safe_content}</div>"
                
                # 更新占位符内容
                self.messages[message_id]["placeholder"].markdown(
                    html_content, 
                    unsafe_allow_html=True
                )
                return True
            except Exception:
                return False
        return False

# 获取向量模型
@st.cache_resource
def get_embeddings():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device}
        embedder = HuggingFaceEmbeddings(
            model_name="liam168/bert-large-chinese",
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        )
        _ = embedder.embed_query("测试")
        return embedder
    except Exception:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 辅助函数：格式化消息
def format_messages(template_messages, **kwargs):
    messages = []
    for msg in template_messages:
        content = msg.content.format(**kwargs)
        if isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": content})
        else:
            messages.append({"role": "user", "content": content})
    return messages

# 获取通义千问 LLM 客户端
def get_llm():
    api_key = os.getenv("TONGYI_API_KEY")
    if not api_key:
        st.error("通义API密钥未配置！")
        return None
    try:
        return ChatTongyi(
            model_name="qwen-plus",
            dashscope_api_key=api_key,
            temperature=0.7,
            streaming=True
        )
    except Exception as e:
        st.error(f"通义模型加载失败: {str(e)}")
        return None

def load_documents() -> List[Document]:
    docs = []
    for file_path in glob.glob(f"{DATA_DIR}/*.pdf"):
        try:
            loader = PDFPlumberLoader(file_path)
            pages = loader.load()
            full_text = "\n".join([p.page_content for p in pages])
            file_name = os.path.basename(file_path)
            docs.append(Document(page_content=full_text, metadata={"source": file_name}))
        except Exception:
            continue
    return docs

# 简单向量数据库
class InMemorySimpleVectorStore:
    def __init__(self, embedding_function, texts, metadatas=None):
        self.embedding_function = embedding_function
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.doc_embeddings = []
        for text in texts:
            try:
                embedding = embedding_function.embed_query(text)
                self.doc_embeddings.append(embedding)
            except Exception:
                self.doc_embeddings.append(np.zeros(768, dtype=np.float32))

    def similarity_search(self, query, k=4):
        query_embedding = self.embedding_function.embed_query(query)
        query_vec = np.array(query_embedding, dtype=np.float32)
        similarities = []
        for doc_embedding in self.doc_embeddings:
            doc_vec = np.array(doc_embedding, dtype=np.float32)
            dot_product = np.dot(query_vec, doc_vec)
            query_norm = np.linalg.norm(query_vec)
            doc_norm = np.linalg.norm(doc_vec)
            similarity = dot_product / (query_norm * doc_norm) if query_norm > 0 and doc_norm > 0 else 0
            similarities.append(similarity)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [Document(page_content=self.texts[i], metadata=self.metadatas[i]) for i in top_indices]

# 加载向量数据库
def load_or_create_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                vector_store = pickle.load(f)
            return vector_store
        except Exception:
            return None
    return None

# 构建向量数据库和索引
@st.cache_resource
def build_vector_store_and_index():
    vector_store = load_or_create_vector_store()
    if vector_store:
        try:
            es = Elasticsearch("http://localhost:9200")
            es.info()
            return vector_store, es
        except Exception:
            return vector_store, None

    docs = load_documents()
    embedder = get_embeddings()
    if not embedder:
        return None, None

    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not docs:
        return None, None

    try:
        try:
            vector_store = FAISS.from_documents(docs, embedder)
        except Exception:
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            vector_store = InMemorySimpleVectorStore(embedder, texts, metadatas)

        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(vector_store, f)

        try:
            es = Elasticsearch("http://localhost:9200")
            es.info()
            if not es.indices.exists(index=ES_INDEX_NAME):
                actions = [{
                    "_index": ES_INDEX_NAME,
                    "_id": i,
                    "_source": {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown")
                    }
                } for i, doc in enumerate(docs)]
                for action in actions:
                    es.index(index=ES_INDEX_NAME, body=action['_source'])
            return vector_store, es
        except Exception:
            return vector_store, None

    except Exception:
        return None, None

# 关键字匹配        
def keyword_search(es, query, top_k=5):
    if not es:
        return []
    try:
        res = es.search(
            index=ES_INDEX_NAME,
            query={"match": {"content": query}},
            size=top_k
        )
        return [Document(page_content=hit["_source"]["content"], metadata={"source": hit["_source"]["source"]}) 
                for hit in res["hits"]["hits"]]
    except Exception:
        return []

# 判断问题是否为科普问题
def is_scientific_query(question: str) -> bool:
    llm = get_llm()
    if not llm:
        return True
    try:
        messages = format_messages(PROMPTS["is_scientific_query"], question=question)
        response = llm.invoke(messages)
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        result = result.strip().lower()
        return result.startswith("是") or "是" in result or result.startswith("yes")
    except Exception:
        return True

# 流式响应生成器
def generate_response_stream(user_input: str, is_sci_question: bool, rag_docs: List[Document] = None) -> Generator[str, None, None]:
    llm = get_llm()
    if not llm:
        yield "系统错误：无法加载语言模型。"
        return
    
    # 构建对话历史字符串
    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" 
         for msg in st.session_state.chat_history]
    )
    
    if is_sci_question and rag_docs:
        # 构建RAG上下文（限制长度）
        context = "\n".join(
            [f"文档({i+1}): {doc.page_content[:500]}" 
             for i, doc in enumerate(rag_docs[:3])]  # 只取前3个文档，限制长度
        )
        
        # 使用多轮对话模板
        messages = format_messages(
            PROMPTS["multi_turn"],
            context=context,
            history=history_str,
            question=user_input
        )
    else:
        # 普通对话使用简单模板
        messages = format_messages(
            PROMPTS["simple_answer"],
            question=user_input
        )
        # 添加历史上下文
        messages.insert(0, {"role": "system", "content": f"对话历史：\n{history_str}"})
    
    try:
        # 使用流式调用模型
        response_stream = llm.stream(messages)
        
        for chunk in response_stream:
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)
            
            yield content
            
    except Exception as e:
        yield f"生成回答时出错：{str(e)}"

# ... 前面的代码保持不变 ...

# ... 前面的代码保持不变 ...

def main():
    # 初始化所有必要的会话状态变量
    required_states = [
        "high_contrast_mode",
        "chat_history",
        "vector_store",
        "es",
        "message_manager"
    ]
    
    for state in required_states:
        if state not in st.session_state:
            # 初始化高对比度模式
            if state == "high_contrast_mode":
                st.session_state.high_contrast_mode = True
            
            # 初始化聊天历史
            elif state == "chat_history":
                st.session_state.chat_history = [
                    {"role": "system", "content": "你是理性且友好的科学问答助手"}
                ]
            
            # 初始化向量数据库
            elif state == "vector_store":
                st.session_state.vector_store = None
            
            # 初始化Elasticsearch
            elif state == "es":
                st.session_state.es = None
            
            # 初始化消息管理器
            elif state == "message_manager":
                st.session_state.message_manager = ChatMessageManager()
    
    # 根据高对比度模式状态选择CSS样式
    if st.session_state.high_contrast_mode:
        css_style = """
        <style>
            body {
                background-color: #000000;
                color: #FFFFFF;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stChatMessage {
                background-color: #000000!important;
            }
            .user-message {
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
                border-left: 4px solid #00FFFF; /* 青色边框 */
                background-color: #111111;
                color: #FFFFFF;
                font-size: 16px;
            }
            .assistant-message {
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
                border-left: 4px solid #FFA500; /* 橙色边框 */
                background-color: #111111;
                color: #FFFFFF;
                font-size: 16px;
            }
            .stTextInput > div > div > input {
                color: #FFFFFF;
                background-color: #222222;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            .stButton > button {
                background-color: #FFA500!important;
                color: #000000!important;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            .stTitle {
                color: #FFFFFF;
                text-align: center;
                font-size: 24px;
                margin-bottom: 20px;
            }
            .toggle-btn {
                background-color: #FFA500!important;
                color: #000000!important;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            /* 去除Streamlit默认元素 */
            .stApp .reportview-container .main .block-container {
                padding-top: 2rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 2rem;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """
    else:
        css_style = """
        <style>
            body {
                background-color: #FFFFFF;
                color: #333333;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stChatMessage {
                background-color: #FFFFFF!important;
            }
            .user-message {
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
                border-left: 4px solid #007BFF;
                background-color: #E6F0FF;
                color: #333333;
                font-size: 16px;
            }
            .assistant-message {
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
                border-left: 4px solid #28A745;
                background-color: #E6F7EB;
                color: #333333;
                font-size: 16px;
            }
            .stTextInput > div > div > input {
                color: #333333;
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            .stButton > button {
                background-color: #007BFF!important;
                color: #FFFFFF!important;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            .stTitle {
                color: #333333;
                text-align: center;
                font-size: 24px;
                margin-bottom: 20px;
            }
            .toggle-btn {
                background-color: #007BFF!important;
                color: #FFFFFF!important;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            /* 去除Streamlit默认元素 */
            .stApp .reportview-container .main .block-container {
                padding-top: 2rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 2rem;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """
    
    st.markdown(css_style, unsafe_allow_html=True)
    
    # 创建容器用于按钮布局
    button_container = st.container()
    with button_container:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.title("🌟 科普知识问答助手")
        with col2:
            if st.session_state.high_contrast_mode:
                button = st.button("退出高对比度模式", key="contrast_toggle", 
                                help="点击返回标准显示模式")
            else:
                button = st.button("进入高对比度模式", key="contrast_toggle", 
                                help="点击切换为高对比度显示")
    
    # 显示界面说明（根据当前模式调整样式）
    if st.session_state.high_contrast_mode:
        st.markdown("""
        <div style="background-color:#111111; padding:15px; border-radius:8px; border-left:4px solid #FFA500; margin-bottom:20px;">
            <p>💡💡 <b>欢迎使用科普知识问答助手！</b></p>
            <p>🔍🔍 请输入您的科学问题，我会尽力提供专业回答</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#E6F7EB; padding:15px; border-radius:8px; border-left:4px solid #28A745; margin-bottom:20px;">
            <p>💡💡 <b>欢迎使用科普知识问答助手！</b></p>
            <p>🔍🔍 请输入您的科学问题，我会尽力提供专业回答</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 初始化向量数据库
    if st.session_state.vector_store is None:
        with st.spinner("正在加载知识库，请稍候..."):
            st.session_state.vector_store, st.session_state.es = build_vector_store_and_index()
            if st.session_state.vector_store is None:
                st.error("⚠️ 向量数据库初始化失败")
            if st.session_state.es is None:
                st.warning("⚠️ Elasticsearch连接失败")
    
    # 显示聊天历史
    for msg in st.session_state.chat_history:
        if msg["role"] != "system":
            # 使用Streamlit内置组件显示消息
            with st.chat_message(msg["role"]):
                st.markdown(msg['content'])
    
    # 用户输入
    user_input = st.chat_input("请输入您的问题~")
    
    if user_input:
        # 添加用户消息
        user_message = {"role": "user", "content": user_input}
        st.session_state.chat_history.append(user_message)
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message'>{user_input}</div>", unsafe_allow_html=True)
        
        # 使用占位符显示思考状态
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(f"""
        <div class='assistant-message' style='text-align:center;'>
            🔍🔍 思考中...
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # 判断问题类型
            is_sci_question = is_scientific_query(user_input)
            rag_docs = []
            
            # 相关文档检索
            if is_sci_question and st.session_state.vector_store:
                try:
                    vector_results = st.session_state.vector_store.similarity_search(user_input, k=3)
                    rag_docs.extend(vector_results)
                    
                    if st.session_state.es:
                        keyword_results = keyword_search(st.session_state.es, user_input, top_k=2)
                        rag_docs.extend(keyword_results)
                        
                    # 去重
                    seen_content = set()
                    unique_docs = []
                    for doc in rag_docs:
                        if doc.page_content not in seen_content:
                            seen_content.add(doc.page_content)
                            unique_docs.append(doc)
                    rag_docs = unique_docs[:3]
                except Exception as e:
                    st.error(f"搜索出错: {str(e)}")
            
            # 生成响应
            response = ""
            response_container = st.empty()
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # 生成响应内容
                generator = generate_response_stream(
                    user_input, 
                    is_sci_question, 
                    rag_docs if rag_docs else None
                )
                
                # 直接流式更新响应占位符
                for chunk in generator:
                    full_response += chunk
                    # 用临时变量解决反斜杠问题
                    temp_response = full_response.replace("\n", "<br>")
                    # 实时更新显示内容（带换行处理）
                    response_placeholder.markdown(
                        f"<div class='assistant-message'>{temp_response}</div>", 
                        unsafe_allow_html=True
                    )
                    # 添加微小延迟确保UI更新
                    time.sleep(0.01)
                
                # 最终确保完整内容显示
                # 再次使用临时变量
                final_response = full_response.replace("\n", "<br>")
                response_placeholder.markdown(
                    f"<div class='assistant-message'>{final_response}</div>", 
                    unsafe_allow_html=True
                )
                
                # 添加到聊天历史
                assistant_message = {"role": "assistant", "content": full_response}
                st.session_state.chat_history.append(assistant_message)
            
            # 清除思考状态
            thinking_placeholder.empty()
        except Exception as e:
            # 错误处理
            error_msg = f"⚠️ 系统出错: {str(e)}"
            
            # 显示错误
            with st.chat_message("assistant"):
                st.markdown(f"<div class='assistant-message'>{error_msg}</div>", unsafe_allow_html=True)
            
            # 添加错误消息到历史
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            # 清除思考状态
            thinking_placeholder.empty()
            
    # 按钮切换处理（放在最后确保其他元素已渲染）
    if button:
        st.session_state.high_contrast_mode = not st.session_state.high_contrast_mode
        st.rerun()

if __name__ == "__main__":
    # 创建缓存目录
    os.makedirs(os.path.join(os.getcwd(), "cache"), exist_ok=True)
    
    # 运行主程序
    main()