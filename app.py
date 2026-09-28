# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import time
import base64

# --- Configuración de la página ---
st.set_page_config(
    page_title="Proyección Comercial | Fashion Retail", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Lógica para Imagen de Fondo y Tema Oscuro Premium ---
def set_background(image_file):
    """Codifica una imagen local a base64 y la inyecta como fondo mediante CSS."""
    try:
        with open(image_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f"""
            <style>
            /* Inyección de la imagen al fondo de la aplicación */
            .stApp {{
                background-image: url(data:image/jpeg;base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            
            /* Capa oscura general para apagar el brillo de la imagen y evitar que se vea pixelada */
            [data-testid="stAppViewContainer"] > .main {{
                background-color: rgba(0, 0, 0, 0.55); 
            }}

            /* Estilo "Dark Glassmorphism" para los paneles y pestañas */
            .stTabs, .metric-container, div[data-testid="stMarkdownContainer"] > p {{
                background-color: rgba(15, 23, 42, 0.85); /* Azul muy oscuro casi negro, con 85% de opacidad */
                padding: 15px;
                border-radius: 12px;
                color: #F8FAFC !important; /* Blanco hueso para evitar fatiga visual */
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            /* Forzar el color de todos los textos principales a blanco */
            h1, h2, h3, label, .stMarkdown {{
                color: #FFFFFF !important;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
            }}

            /* Estilo del botón principal */
            .stButton>button {{
                width: 100%;
                background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
                color: white;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                transition: all 0.3s ease;
            }}
            .stButton>button:hover {{
                border-color: #94A3B8;
                box-shadow: 0 0 15px rgba(255,255,255,0.2);
                transform: translateY(-2px);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ No se encontró la imagen de fondo. Verifique que 'fondo_despliegue.jfif' esté en el repositorio.")

# Llamada a la función con el nombre exacto de su archivo
set_background('fondo_despliegue.jfif')

# --- Carga de artefactos ---
@st.cache_resource
def cargar_modelo():
    filename = 'modelo-gboosting-reg.pkl'
    with open(filename, 'rb') as f:
        modelo_gbc, variables_entrenamiento, min_max_scaler = pickle.load(f)
    return modelo_gbc, variables_entrenamiento, min_max_scaler

modelo_gbc, variables_entrenamiento, min_max_scaler = cargar_modelo()

def formatear_cop(valor):
    """Formatea a pesos colombianos: punto para miles, coma para decimales."""
    return f"${valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- Interfaz Principal ---
st.title("✨ Proyección Comercial por Asesor")
st.markdown("Anticipe el rendimiento mensual combinando historial de ventas, perfil del asesor y contexto de la tienda.")
st.write("---")

tab1, tab2, tab3 = st.tabs(["👤 Perfil del Asesor", "🏬 Contexto de Tienda", "📊 Historial de Ventas (T-3 a T-1)"])

with tab1:
    col1, col2, col3 = st.columns(3)
    edad_asesor = col1.slider('Edad del asesor', min_value=14.0, max_value=60.0, value=29.0, step=1.0)
    genero_asesor = col2.selectbox('Género', ['F', 'M'])
    nacionalidad_asesor = col3.selectbox('Nacionalidad', ['COLOMBIA', 'EXTRANJERO', 'VENEZUELA'])
    
    col4, col5 = st.columns(2)
    tipo_vinculacion = col4.selectbox('Tipo de vinculación', ['INDEF', 'OBRA'])
    total_dias_absentismo = col5.slider('Días de absentismo (mes actual)', min_value=0.0, max_value=31.0, value=0.0, step=1.0)

with tab2:
    col1, col2 = st.columns(2)
    marca = col1.selectbox('Marca', ['MARCA_A', 'MARCA_B', 'MARCA_C', 'MARCA_D'])
    zona_comercial = col2.selectbox('Zona comercial', ['MEDELLIN', 'BOGOTA', 'CALI', 'CARTAGENA', 'BARRANQUILLA', 'OTRA_ZONA'])
    
    col3, col4, col5 = st.columns(3)
    tipo_ubicacion_tienda = col3.selectbox('Ubicación de tienda', ['CENTRO COMERCIAL', 'EXTERIOR'])
    cantidad_cajas = col4.slider('Cantidad de cajas', min_value=1.0, max_value=4.0, value=2.0, step=1.0)
    metros_cuadrados_tienda = col5.number_input('Área de la tienda (m²)', min_value=50.0, max_value=350.0, value=150.0)
    mes_venta = st.selectbox('Mes a proyectar (T)', ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE'])

with tab3:
    st.markdown("Registre las ventas de los últimos 3 meses para capturar la tendencia de la temporada.")
    col1, col2, col3 = st.columns(3)
    venta_t_3 = col1.number_input('Venta mes T-3 (COP)', min_value=0, value=5000000, step=500000, format="%d")
    venta_t_2 = col2.number_input('Venta mes T-2 (COP)', min_value=0, value=5500000, step=500000, format="%d")
    venta_t_1 = col3.number_input('Venta mes T-1 (COP)', min_value=0, value=6000000, step=500000, format="%d")

st.write("---")

# --- Lógica de Inferencia ---
if st.button('Generar Proyección de Ventas'):
    with st.spinner('Analizando variables y calculando pronóstico...'):
        time.sleep(0.6) 
        
        columnas_entrada = ['genero_asesor', 'edad_asesor', 'tipo_vinculacion', 'nacionalidad_asesor',
                            'total_dias_absentismo', 'tipo_ubicacion_tienda', 'marca', 'zona_comercial',
                            'cantidad_cajas', 'metros_cuadrados_tienda', 'mes_venta', 'venta_t_1',
                            'venta_t_2', 'venta_t_3']

        datos = [[genero_asesor, edad_asesor, tipo_vinculacion, nacionalidad_asesor, total_dias_absentismo,
                  tipo_ubicacion_tienda, marca, zona_comercial, cantidad_cajas, metros_cuadrados_tienda,
                  mes_venta, venta_t_1, venta_t_2, venta_t_3]]
        
        data = pd.DataFrame(datos, columns=columnas_entrada)

        columnas_cat = ['genero_asesor', 'tipo_vinculacion', 'nacionalidad_asesor', 'tipo_ubicacion_tienda', 'marca', 'zona_comercial', 'mes_venta']
        data[columnas_cat] = data[columnas_cat].apply(lambda x: x.astype(str).str.upper())

        data_preparada = pd.get_dummies(data, columns=columnas_cat, drop_first=False, dtype=int)
        data_preparada = data_preparada.reindex(columns=variables_entrenamiento, fill_value=0)

        data_preparada['valor_venta_asesor_mes_t'] = 0.0
        col_numericas = ['edad_asesor', 'total_dias_absentismo', 'cantidad_cajas', 'metros_cuadrados_tienda', 'venta_t_1', 'venta_t_2', 'venta_t_3', 'valor_venta_asesor_mes_t']
        data_preparada[col_numericas] = min_max_scaler.transform(data_preparada[col_numericas])

        X_inferencia = data_preparada.drop(columns=['valor_venta_asesor_mes_t'])
        
        # Inferencia estricta con Gradient Boosting (según su requerimiento)
        Y_pred_normalizado = modelo_gbc.predict(X_inferencia)

        temp_array = np.zeros((1, len(col_numericas)))
        indice_target = col_numericas.index('valor_venta_asesor_mes_t')
        temp_array[0, indice_target] = Y_pred_normalizado[0]
        Y_pred_pesos = min_max_scaler.inverse_transform(temp_array)[0, indice_target]

        # --- Visualización de Resultados ---
        st.success("Proyección calculada exitosamente.")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric(label="Venta Estimada (Mes T)", value=formatear_cop(Y_pred_pesos), delta=formatear_cop(Y_pred_pesos - venta_t_1))
            st.caption("Modelo Gradient Boosting (MAE estimado: ± 3.3%)")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            df_tendencia = pd.DataFrame({
                "Mes": ["T-3", "T-2", "T-1", "T (Proyectado)"],
                "Ventas": [venta_t_3, venta_t_2, venta_t_1, Y_pred_pesos]
            }).set_index("Mes")
            st.line_chart(df_tendencia)
