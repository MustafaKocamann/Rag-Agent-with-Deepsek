import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
from pathlib import Path

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern SaaS UI Styling
st.markdown("""
    <style>
    /* Global App Styling */
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);
        color: #E8EAF6;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #E8EAF6 !important;
        border: 2px solid #475569 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput input:focus {
        border-color: #667EEA !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 15px !important;
        padding: 1.2rem !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 15px !important;
        padding: 1.2rem !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px);
    }
    
    /* Text Color */
    .stChatMessage p, .stChatMessage div {
        color: #E8EAF6 !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667EEA;
        background: rgba(30, 41, 59, 0.8);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Headers */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #667EEA !important;
        font-weight: 600 !important;
    }
    
    /* Success/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border-left: 4px solid #10B981 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%) !important;
        border-left: 4px solid #3B82F6 !important;
        border-radius: 8px !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #667EEA !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
        color: #E8EAF6 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #667EEA 0%, #764BA2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667EEA !important;
    }
    </style>
    """, unsafe_allow_html=True)


PROMPT_TEMPLATE = """
Sen, yalnızca sağlanan dokümanlara dayanarak yanıt veren titiz bir araştırma asistanısın. 
Yanıtlarını oluştururken doküman dışına çıkma; eğer aranan bilgi bağlam içerisinde mevcut değilse, 
dış bilgini kullanmak yerine 'Bu bilgi sağlanan dökümanlarda bulunmamaktadır' şeklinde belirt. 
Cevaplarını olabildiğince öz, teknik doğruluğu yüksek ve maksimum 3 cümle olacak şekilde yapılandır.

Bağlam: {document_context}
Soru: {user_query}
Cevap:
"""

# --- YAPILANDIRMA ---
PDF_STORAGE_PATH = "document_store/pdfs/"
EMBEDDING_MODEL_NAME = "mxbai-embed-large"
LLM_MODEL_NAME = "deepseek-r1:8b"

# --- SESSION STATE YÖNETİMİ (KRİTİK!) ---
# Vector DB'yi session_state içinde sakla ki sayfa yenilenince silinmesin
if "vector_db" not in st.session_state:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    st.session_state.vector_db = InMemoryVectorStore(embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

## PDF Dosyası Kaydetme Fonksiyonu
def save_uploaded_file(uploaded_file):
    try:
        Path(PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"❌ Dosya kaydetme hatası: {str(e)}")
        return None

## PDF İşleme Fonksiyonu (Tek fonksiyonda birleştirildi)
def process_pdf(uploaded_file):
    try:
        # Dosyayı kaydet
        file_path = save_uploaded_file(uploaded_file)
        if not file_path:
            return None
        
        # PDF'i yükle
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        if not docs:
            st.error("❌ PDF içeriği okunamadı!")
            return None
        
        # Metni parçala
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        
        # Session state'deki veritabanına ekle (KRİTİK!)
        st.session_state.vector_db.add_documents(chunks)
        
        return len(chunks)
    except Exception as e:
        st.error(f"❌ İşleme hatası: {str(e)}")
        return None

## İlgili Belgeleri Bulma Fonksiyonu
def find_related_documents(query, k=10):
    try:
        # Session state'deki DB'yi kullan (KRİTİK!)
        results = st.session_state.vector_db.similarity_search(query, k=k)
        return results
    except Exception as e:
        st.error(f"❌ Arama hatası: {str(e)}")
        return []

## Cevap Üretme Fonksiyonu
def generate_answer(user_query, context_documents):
    try:
        if not context_documents:
            return "⚠️ Bu bilgi sağlanan dökümanlarda bulunmamaktadır."
        
        context_text = "\n\n".join([doc.page_content for doc in context_documents])
        
        # LLM'i her seferinde oluştur (hafif işlem)
        llm = OllamaLLM(model=LLM_MODEL_NAME)
        conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        response_chain = conversation_prompt | llm
        
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})
    except Exception as e:
        st.error(f"❌ Cevap üretme hatası: {str(e)}")
        return "Üzgünüm, cevap oluştururken bir hata meydana geldi."


# --- HEADER ---
st.title("📚 DocuMind AI")
st.markdown("### 🚀 Gelişmiş Doküman Analiz Asistanı")

# --- SIDEBAR: DOSYA YÜKLEMİ VE KONTROLLER ---
with st.sidebar:
    st.markdown("## 📂 Dosya Merkezi")
    
    uploaded_pdf = st.file_uploader(
        "PDF Yükle",
        type="pdf",
        help="📎 Analiz için PDF dokümanı yükleyin",
        accept_multiple_files=False
    )
    
    if uploaded_pdf:
        # Sadece yeni dosya yüklendiğinde işle
        if st.session_state.current_file != uploaded_pdf.name:
            with st.spinner("🧠 Doküman analiz ediliyor..."):
                chunks_count = process_pdf(uploaded_pdf)
                
                if chunks_count:
                    st.session_state.current_file = uploaded_pdf.name
                    st.session_state.chunks_count = chunks_count
                    st.success(f"✅ Başarılı! {chunks_count} parça indexlendi.")
                    st.balloons()
        else:
            st.info(f"📄 **{uploaded_pdf.name}** yüklendi")
            st.caption(f"📊 {st.session_state.chunks_count} parça indexlendi")
    
    st.markdown("---")
    
    # Sohbeti temizle butonu
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Veritabanını sıfırla butonu
    if st.button("🔄 Veritabanını Sıfırla", use_container_width=True):
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        st.session_state.vector_db = InMemoryVectorStore(embeddings)
        st.session_state.current_file = None
        st.session_state.chunks_count = 0
        st.session_state.messages = []
        st.rerun()

# --- ANA SOHBET ALANI ---
st.markdown("### 💬 Sohbet")

# Yüklü doküman yoksa uyarı göster
if not st.session_state.current_file:
    st.info("👈 Lütfen sol panelden bir PDF dokümanı yükleyin")
    st.markdown("""
    ### 🌟 Özellikler
    - 🔍 **Semantik Arama**: Anlamlı bilgileri anında bulun
    - 🎯 **Hassas Cevaplar**: Dokümanınıza dayalı doğru yanıtlar alın
    - 💡 **Bağlam Farkındalığı**: Sorularınızın tam bağlamını anlar
    - ⚡ **Hızlı İşleme**: DeepSeek AI modeli ile güçlendirilmiş
    """)
else:
    # Sohbet geçmişini göster
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
    
    # Sohbet girişi
    if prompt := st.chat_input("💭 Belgeniz hakkında bir şey sorun..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Asistan yanıtı
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Dokümanlar taranıyor..."):
                # Session state'deki DB'de arama yap
                related_docs = find_related_documents(prompt, k=10)
                
                # Debug bilgisi
                with st.expander("📊 Arama Sonuçları", expanded=False):
                    st.info(f"**Bulunan parça sayısı:** {len(related_docs)}")
                    if related_docs:
                        for i, doc in enumerate(related_docs[:3], 1):
                            st.caption(f"**Sonuç {i}:**")
                            st.text(doc.page_content[:200] + "...")
                
                # Cevap üret
                response = generate_answer(prompt, related_docs)
                st.markdown(response)
                
                # Asistan mesajını kaydet
                st.session_state.messages.append({"role": "assistant", "content": response})