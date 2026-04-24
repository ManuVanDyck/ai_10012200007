import streamlit as st
from sentence_transformers import SentenceTransformer
from embeddings import load_chroma_collection
from pipeline import end_to_end_pipeline


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_collection():
    return load_chroma_collection()


if 'recent_queries' not in st.session_state:
    st.session_state['recent_queries'] = []

st.set_page_config(page_title='GovNet AI', page_icon='🤖', layout='wide')

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #e6f0ff 0%, #fff8c8 100%);
        color: #022d5b;
    }
    div[data-testid='stSidebar'] {
        background: linear-gradient(180deg, #002b5c 0%, #0048a6 100%);
        color: #ffffff;
    }
    .css-1offfwp {
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        border-radius: 1.2rem;
    }
    .stButton>button {
        background-color: #0047ab;
        color: #ffffff;
        border-radius: 1rem;
        border: 1px solid #003775;
        padding: 0.8rem 1.2rem;
    }
    .stButton>button:hover {
        background-color: #003775;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        border: 2px solid #ffd700 !important;
        border-radius: 0.8rem;
        padding: 0.75rem 1rem;
    }
    .stExpander>div {
        background-color: #fff9d9;
        border: 1px solid #ffe676;
        border-radius: 1rem;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #002b5c;
    }
    .result-card {
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 1rem;
        border: 1px solid #dfe6ff;
        padding: 1.2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .metric-pill {
        display: inline-block;
        margin: 0.15rem 0.25rem 0.15rem 0;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background-color: #ffe477;
        color: #002b5c;
        font-weight: 600;
    }
    .footer {
        color: #002b5c;
        font-size: 0.95rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0, 45, 91, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('### GovNet AI')
    st.write('Your smart assistant for government data, policy insight, and election-context retrieval.')
    st.markdown('---')
    st.markdown('#### How to use')
    st.markdown(
        '''
- Type a question about government data or elections.
- Review retrieved context and the generated answer.
- Use the prompt expander to inspect the exact LLM input.
        '''
    )
    st.markdown('---')
    st.markdown('#### Example questions')
    st.write('• What were the voter turnout trends?')
    st.write('• Summarize the key election results.')
    st.write('• Which regions showed the strongest support?')
    st.markdown('---')
    st.markdown('#### Data source')
    st.write('CSV and report chunks from government and election datasets.')
    st.markdown('---')
    st.markdown('#### Recent queries')
    if st.session_state['recent_queries']:
        for item in reversed(st.session_state['recent_queries'][-5:]):
            st.write(f'- {item}')
    else:
        st.write('No recent queries yet.')

st.title('GovNet AI')
st.markdown('### Welcome to GovNet AI — your intelligent government data assistant.')
st.markdown(
    'Ask a question about policy, election data, or government reports and get a grounded answer with supporting context.'
)

query = st.text_input('Ask a question...', '')
run_query = st.button('Ask GovNet AI')

if run_query:
    if not query.strip():
        st.warning('Please enter a question to continue.')
    else:
        st.session_state['recent_queries'].append(query)
        with st.spinner('Searching relevant documents and generating an answer...'):
            model = load_model()
            collection = load_collection()
            result = end_to_end_pipeline(query, collection, model, prompt_version='v4', display=False)

        top_similarity = max(result['similarity_scores']) if result['similarity_scores'] else 0.0
        top_combined = max(result['combined_scores']) if result['combined_scores'] else 0.0
        chunk_count = len(result['retrieved_documents'])

        st.markdown('### Search summary')
        st.markdown(
            f"<div class='result-card'>"
            f"<span class='metric-pill'>Chunks retrieved: {chunk_count}</span>"
            f"<span class='metric-pill'>Top similarity: {top_similarity:.4f}</span>"
            f"<span class='metric-pill'>Top combined score: {top_combined:.4f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('## Retrieved Context')
        if result['retrieved_documents']:
            for i, (doc, sim, dom, key, comb) in enumerate(zip(
                result['retrieved_documents'],
                result['similarity_scores'],
                result['domain_scores'],
                result['keyword_scores'],
                result['combined_scores']), start=1):
                with st.expander(f'Context chunk {i} — Combined: {comb:.4f}'):
                    st.markdown(
                        f"<div class='result-card'>"
                        f"<p><strong>Similarity:</strong> {sim:.4f}  |  <strong>Domain:</strong> {dom:.4f}  |  <strong>Keyword:</strong> {key:.4f}  |  <strong>Combined:</strong> {comb:.4f}</p>"
                        f"<p>{doc}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info('No relevant context was found for that question.')

        st.markdown('## Answer')
        st.markdown(
            f"<div class='result-card'><p>{result['response']}</p></div>",
            unsafe_allow_html=True,
        )

        with st.expander('Show the prompt sent to the LLM'):
            st.code(result['prompt'])

        st.markdown('<div class="footer">Built for smarter government data workflows with GovNet AI.</div>', unsafe_allow_html=True)
