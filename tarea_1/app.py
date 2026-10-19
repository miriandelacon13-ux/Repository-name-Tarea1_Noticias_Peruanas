import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Clasificador de Noticias Peruanas",
    page_icon="📰",
    layout="wide"
)

# =========================
# ESTILOS
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #173F5F;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #5B6B7A;
    text-align: center;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 5px solid #3B82F6;
}

.card-title {
    color: #64748B;
    font-size: 15px;
    margin-bottom: 8px;
}

.card-value {
    color: #0F172A;
    font-size: 28px;
    font-weight: 700;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #173F5F;
    margin-top: 30px;
    margin-bottom: 15px;
}

div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)
@st.cache_resource
def cargar_modelo():
    modelo_hf = "Mirian13/clasificador-noticias-peruanas"

    tokenizer = AutoTokenizer.from_pretrained(modelo_hf)

    model = AutoModelForSequenceClassification.from_pretrained(
        modelo_hf
    )

    model.eval()

    return tokenizer, model

tokenizer, model = cargar_modelo()

# =========================
# CARGA DE DATOS
# =========================

df = pd.read_csv(BASE_DIR / "datos" / "noticias.csv")

# =========================
# ENCABEZADO
# =========================

st.markdown(
    '<div class="main-title">📰 Noticias peruanas: del titular al clasificador</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Análisis y clasificación automática de noticias de El Comercio</div>',
    unsafe_allow_html=True
)

# =========================
# MÉTRICAS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Total de noticias</div>
        <div class="card-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Medio</div>
        <div class="card-value">{df["medio"].iloc[0]}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Categorías</div>
        <div class="card-value">{df["etiqueta"].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-title">F1 Macro final</div>
        <div class="card-value">66.56%</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# GRÁFICOS
# =========================

st.markdown(
    '<div class="section-title">📊 Distribución de noticias</div>',
    unsafe_allow_html=True
)

conteo = df["etiqueta"].value_counts().reset_index()
conteo.columns = ["Categoría", "Cantidad"]

fig = px.bar(
    conteo,
    x="Categoría",
    y="Cantidad",
    color="Categoría",
    text="Cantidad",
    title="Cantidad de noticias por categoría"
)

fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    title_font_size=20,
    showlegend=False
)

fig.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# CLASIFICADOR
# =========================
st.markdown(
    '<div class="section-title">🤖 Clasifica un titular</div>',
    unsafe_allow_html=True
)

st.markdown("""
Escribe un titular de noticia y el modelo identificará automáticamente
si pertenece a **política, economía, deportes o ciencia y tecnología**.
""")

titular_usuario = st.text_input(
    "Escribe un titular:",
    placeholder="Ejemplo: Gobierno anuncia nuevas medidas económicas..."
)

if st.button("🔍 Clasificar noticia", use_container_width=True):

    if tokenizer is None or model is None:
        st.error(
            "El modelo entrenado no está disponible en este entorno. "
            "Los archivos pesados del modelo no se incluyen en GitHub."
        )
        st.stop()

    if titular_usuario.strip():

        entrada = tokenizer(
            titular_usuario,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            salida = model(**entrada)

        probabilidades = torch.softmax(
            salida.logits,
            dim=-1
        )[0]

        prediccion = torch.argmax(
            probabilidades
        ).item()

        categoria = model.config.id2label[prediccion]

        confianza = (
            probabilidades[prediccion].item() * 100
        )

        st.success(
            f"Categoría predicha: **{categoria.replace('_', ' ').title()}**"
        )

        st.progress(
            int(confianza)
        )

        st.write(
            f"**Confianza del modelo: {confianza:.2f}%**"
        )

    else:
        st.warning("Escribe un titular antes de clasificar.")

st.markdown(
    '<div class="section-title">📈 Comparación de modelos</div>',
    unsafe_allow_html=True
)

comparacion = pd.DataFrame({
    "Modelo": [
        "Zero-shot",
        "Ollama Qwen 2.5 3B",
        "DistilBERT fine-tuned"
    ],
    "Accuracy": [
        65.00,
        73.33,
        64.81
    ],
    "F1 Macro": [
        63.28,
        73.10,
        66.56
    ]
})


fig_modelos = px.bar(
    comparacion,
    x="Modelo",
    y=["Accuracy", "F1 Macro"],
    barmode="group",
    text_auto=".2f",
    title="Rendimiento de los modelos (%)"
)

fig_modelos.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Porcentaje",
    legend_title="Métrica"
)

fig_modelos.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_modelos,
    use_container_width=True
)


st.markdown(
    '<div class="section-title">📋 Noticias recolectadas</div>',
    unsafe_allow_html=True
)

categorias = ["Todas"] + sorted(df["etiqueta"].unique().tolist())

categoria_seleccionada = st.selectbox(
    "Filtrar por categoría",
    categorias
)

if categoria_seleccionada != "Todas":
    df_mostrar = df[
        df["etiqueta"] == categoria_seleccionada
    ]
else:
    df_mostrar = df

st.dataframe(
    df_mostrar,
    use_container_width=True,
    hide_index=True
)