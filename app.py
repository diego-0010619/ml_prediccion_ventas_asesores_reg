# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import time

# --- Configuración de la página ---
st.set_page_config(
    page_title="Proyección Comercial | Fashion Retail", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Estilos CSS personalizados para una UI premium ---
st.markdown("""
    <style>
    .main {background-color: #FAFAFA;}
    h1 {color: #1E1E1E; font-family: 'Helvetica Neue', sans-serif; font-weight: 300; letter-spacing: 1px;}
    .stButton>button {
        width: 100%;
        background-color: #1E1E1E;
        color: white;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4A4A4A;
        border-color: #4A4A4A;
        transform: translateY(-2px);
    }
    .metric-container {
        padding: 20px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Carga de artefactos ---
@st.cache_resource
def cargar_modelo():
    """Carga el modelo en caché para evitar lectura en disco con cada interacción de la UI."""
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

# Organización en pestañas para reducir carga visual
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
    venta_t_3 = col1.number_input('Venta mes T-3 (COP)', min_value=0.0, value=5000000.0, step=500000.0)
    venta_t_2 = col2.number_input('Venta mes T-2 (COP)', min_value=0.0, value=5500000.0, step=500000.0)
    venta_t_1 = col3.number_input('Venta mes T-1 (COP)', min_value=0.0, value=6000000.0, step=500000.0)

st.write("---")

# --- Lógica de Inferencia ---
if st.button('Generar Proyección de Ventas'):
    with st.spinner('Analizando variables y calculando pronóstico...'):
        time.sleep(0.6) # Simula transición suave
        
        # 1. Empaquetar captura
        columnas_entrada = ['genero_asesor', 'edad_asesor', 'tipo_vinculacion', 'nacionalidad_asesor',
                            'total_dias_absentismo', 'tipo_ubicacion_tienda', 'marca', 'zona_comercial',
                            'cantidad_cajas', 'metros_cuadrados_tienda', 'mes_venta', 'venta_t_1',
                            'venta_t_2', 'venta_t_3']

        datos = [[genero_asesor, edad_asesor, tipo_vinculacion, nacionalidad_asesor, total_dias_absentismo,
                  tipo_ubicacion_tienda, marca, zona_comercial, cantidad_cajas, metros_cuadrados_tienda,
                  mes_venta, venta_t_1, venta_t_2, venta_t_3]]
        
        data = pd.DataFrame(datos, columns=columnas_entrada)

        # 2. Preprocesamiento Vectorial
        columnas_cat = ['genero_asesor', 'tipo_vinculacion', 'nacionalidad_asesor', 'tipo_ubicacion_tienda', 'marca', 'zona_comercial', 'mes_venta']
        data[columnas_cat] = data[columnas_cat].apply(lambda x: x.astype(str).str.upper())

        data_preparada = pd.get_dummies(data, columns=columnas_cat, drop_first=False, dtype=int)
        data_preparada = data_preparada.reindex(columns=variables_entrenamiento, fill_value=0)

        # 3. Escalado
        data_preparada['valor_venta_asesor_mes_t'] = 0.0
        col_numericas = ['edad_asesor', 'total_dias_absentismo', 'cantidad_cajas', 'metros_cuadrados_tienda', 'venta_t_1', 'venta_t_2', 'venta_t_3', 'valor_venta_asesor_mes_t']
        data_preparada[col_numericas] = min_max_scaler.transform(data_preparada[col_numericas])

        # 4. Inferencia
        X_inferencia = data_preparada.drop(columns=['valor_venta_asesor_mes_t'])
        Y_pred_normalizado = modelo_gbc.predict(X_inferencia)

        # 5. Transformación Inversa
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
            st.caption("MAE estimado: ± 3.3%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            # Gráfico de tendencia para la vista ejecutiva
            df_tendencia = pd.DataFrame({
                "Mes": ["T-3", "T-2", "T-1", "T (Proyectado)"],
                "Ventas": [venta_t_3, venta_t_2, venta_t_1, Y_pred_pesos]
            }).set_index("Mes")
            st.line_chart(df_tendencia, color="#1E1E1E")
