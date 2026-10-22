# -*- coding: utf-8 -*-
"""
Retail Intelligence Platform | Fashion Stores
Despliegue y Simulación Comercial de Ventas por Asesor
Modelo Predictivo: Gradient Boosting Regressor (Scikit-Learn)
"""

import base64
import time
import pickle
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ==============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Retail Intelligence | Proyección de Ventas",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DEFINICIÓN DE PERFILES RÁPIDOS (PRESETS PARA PRUEBAS EN 1 CLIC)
# ==============================================================================
PRESETS = {
    "🚀 Asesor Consolidado (Estabilidad Comercial)": {
        "edad_asesor": 28.0,
        "genero_asesor": "M",
        "nacionalidad_asesor": "COLOMBIA",
        "tipo_vinculacion": "INDEF",
        "total_dias_absentismo": 1.0,
        "marca": "MARCA_B",
        "zona_comercial": "MEDELLIN",
        "tipo_ubicacion_tienda": "CENTRO COMERCIAL",
        "cantidad_cajas": 2.0,
        "metros_cuadrados_tienda": 140.0,
        "mes_venta": "OCTUBRE",
        "venta_t_3": 6000000,
        "venta_t_2": 6500000,
        "venta_t_1": 7200000
    },
    "🌟 Asesor Estrella (Flagship CC - Temporada Alta)": {
        "edad_asesor": 31.0,
        "genero_asesor": "F",
        "nacionalidad_asesor": "COLOMBIA",
        "tipo_vinculacion": "INDEF",
        "total_dias_absentismo": 0.0,
        "marca": "MARCA_A",
        "zona_comercial": "BOGOTA",
        "tipo_ubicacion_tienda": "CENTRO COMERCIAL",
        "cantidad_cajas": 3.0,
        "metros_cuadrados_tienda": 220.0,
        "mes_venta": "DICIEMBRE",
        "venta_t_3": 12500000,
        "venta_t_2": 14000000,
        "venta_t_1": 16800000
    },
    "💼 Asesor Junior / Nuevo Ingreso": {
        "edad_asesor": 21.0,
        "genero_asesor": "F",
        "nacionalidad_asesor": "COLOMBIA",
        "tipo_vinculacion": "OBRA",
        "total_dias_absentismo": 0.0,
        "marca": "MARCA_C",
        "zona_comercial": "CALI",
        "tipo_ubicacion_tienda": "EXTERIOR",
        "cantidad_cajas": 2.0,
        "metros_cuadrados_tienda": 95.0,
        "mes_venta": "MARZO",
        "venta_t_3": 3200000,
        "venta_t_2": 3800000,
        "venta_t_1": 4200000
    },
    "⚠️ Asesor con Alto Absentismo (Alerta de Riesgo)": {
        "edad_asesor": 35.0,
        "genero_asesor": "M",
        "nacionalidad_asesor": "COLOMBIA",
        "tipo_vinculacion": "INDEF",
        "total_dias_absentismo": 12.0,
        "marca": "MARCA_D",
        "zona_comercial": "BARRANQUILLA",
        "tipo_ubicacion_tienda": "CENTRO COMERCIAL",
        "cantidad_cajas": 2.0,
        "metros_cuadrados_tienda": 130.0,
        "mes_venta": "MAYO",
        "venta_t_3": 7500000,
        "venta_t_2": 5400000,
        "venta_t_1": 4100000
    }
}

# Inicialización segura de claves en Session State
def inicializar_estado():
    preset_default = PRESETS["🚀 Asesor Consolidado (Estabilidad Comercial)"]
    for campo, valor in preset_default.items():
        clave = f"inp_{campo}"
        if clave not in st.session_state:
            st.session_state[clave] = valor
    if "preset_actual" not in st.session_state:
        st.session_state.preset_actual = "🚀 Asesor Consolidado (Estabilidad Comercial)"

inicializar_estado()

def callback_cambio_preset():
    elegido = st.session_state.selector_preset
    if elegido in PRESETS:
        vals = PRESETS[elegido]
        for k, v in vals.items():
            st.session_state[f"inp_{k}"] = v
        st.session_state.preset_actual = elegido
        if 'ultima_prediccion' in st.session_state:
            del st.session_state['ultima_prediccion']

# ==============================================================================
# CARGA Y CACHÉ DEL MODELO
# ==============================================================================
@st.cache_resource(show_spinner=False)
def cargar_artefactos():
    """Carga el modelo Gradient Boosting, variables de entrenamiento y escalador."""
    filename = 'modelo-gboosting-reg.pkl'
    try:
        with open(filename, 'rb') as f:
            modelo_gbc, variables_entrenamiento, min_max_scaler = pickle.load(f)
        return modelo_gbc, variables_entrenamiento, min_max_scaler, None
    except Exception as e:
        return None, None, None, str(e)

modelo_gbc, variables_entrenamiento, min_max_scaler, error_carga = cargar_artefactos()

# ==============================================================================
# FUNCIONES DE INFERENCIA Y FORMATEO
# ==============================================================================
def formatear_cop(valor):
    """Formatea valores monetarios a pesos colombianos (COP)."""
    if valor is None or np.isnan(valor):
        return "$0"
    return f"${valor:,.0f}".replace(",", ".")

def predecir_venta(datos_dict):
    """
    Inferencia del modelo Gradient Boosting con preprocesamiento riguroso:
    One-Hot Encoding -> Reindexación con 38 columnas -> Escalado MinMax -> Desescalado a COP.
    """
    columnas_entrada = [
        'genero_asesor', 'edad_asesor', 'tipo_vinculacion', 'nacionalidad_asesor',
        'total_dias_absentismo', 'tipo_ubicacion_tienda', 'marca', 'zona_comercial',
        'cantidad_cajas', 'metros_cuadrados_tienda', 'mes_venta', 'venta_t_1',
        'venta_t_2', 'venta_t_3'
    ]
    
    fila = [datos_dict[col] for col in columnas_entrada]
    df_raw = pd.DataFrame([fila], columns=columnas_entrada)
    
    columnas_cat = ['genero_asesor', 'tipo_vinculacion', 'nacionalidad_asesor', 'tipo_ubicacion_tienda', 'marca', 'zona_comercial', 'mes_venta']
    df_raw[columnas_cat] = df_raw[columnas_cat].apply(lambda x: x.astype(str).str.upper())
    
    df_dummies = pd.get_dummies(df_raw, columns=columnas_cat, drop_first=False, dtype=int)
    df_preparado = df_dummies.reindex(columns=variables_entrenamiento, fill_value=0)
    
    df_preparado['valor_venta_asesor_mes_t'] = 0.0
    col_numericas = [
        'edad_asesor', 'total_dias_absentismo', 'cantidad_cajas', 'metros_cuadrados_tienda',
        'venta_t_1', 'venta_t_2', 'venta_t_3', 'valor_venta_asesor_mes_t'
    ]
    
    df_preparado[col_numericas] = min_max_scaler.transform(df_preparado[col_numericas])
    X_inf = df_preparado.drop(columns=['valor_venta_asesor_mes_t'])
    
    y_pred_norm = modelo_gbc.predict(X_inf)
    
    temp = np.zeros((1, len(col_numericas)))
    idx_target = col_numericas.index('valor_venta_asesor_mes_t')
    temp[0, idx_target] = y_pred_norm[0]
    y_pred_cop = min_max_scaler.inverse_transform(temp)[0, idx_target]
    
    return max(0.0, float(y_pred_cop))

# ==============================================================================
# ESTILOS CSS - DARK LUXURY FASHION RETAIL
# ==============================================================================
def aplicar_estilos(activar_fondo_imagen=False):
    css_fondo = ""
    if activar_fondo_imagen:
        try:
            with open("fondo_despliegue.jfif", "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            css_fondo = f"""
            .stApp {{
                background-image: linear-gradient(rgba(7, 11, 20, 0.88), rgba(4, 7, 15, 0.94)), url(data:image/jpeg;base64,{img_b64});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            """
        except Exception:
            pass

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: radial-gradient(circle at 15% 15%, #0f172a 0%, #070d18 55%, #03060c 100%);
            color: #E2E8F0;
        }}

        {css_fondo}

        /* Títulos */
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
            letter-spacing: -0.02em;
            color: #FFFFFF !important;
        }}

        .gradient-title {{
            background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 45%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
            line-height: 1.2;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 700;
            color: #38BDF8;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        /* Tarjetas Glassmorphism */
        .glass-card {{
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.5);
            margin-bottom: 14px;
        }}

        /* KPI Card Principal */
        .kpi-card {{
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.88) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 20px 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 12px 30px -10px rgba(56, 189, 248, 0.18);
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #38BDF8, #818CF8, #F43F5E);
        }}

        .kpi-label {{
            font-size: 0.82rem;
            color: #94A3B8;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}

        .kpi-value {{
            font-size: 2.05rem;
            font-weight: 800;
            color: #FFFFFF;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }}

        .kpi-sub {{
            font-size: 0.82rem;
            color: #CBD5E1;
            margin-top: 8px;
            font-weight: 500;
        }}

        .pill-positive {{
            color: #34D399;
            background: rgba(52, 211, 153, 0.16);
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
        }}

        .pill-negative {{
            color: #F87171;
            background: rgba(248, 113, 113, 0.16);
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
        }}

        /* Estilo de pestañas */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: rgba(15, 23, 42, 0.65);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.07);
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 42px;
            border-radius: 8px;
            font-weight: 600;
            color: #94A3B8;
            padding: 0 16px;
            border: none;
            transition: all 0.2s ease;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(56, 189, 248, 0.45) !important;
        }}

        /* Botón de Ejecución */
        .stButton>button {{
            background: linear-gradient(135deg, #0284C7 0%, #2563EB 50%, #4F46E5 100%);
            color: #FFFFFF;
            font-weight: 700;
            font-size: 1.05rem;
            letter-spacing: 0.04em;
            padding: 14px 28px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
        }}

        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 32px -5px rgba(37, 99, 235, 0.7);
            border-color: rgba(255, 255, 255, 0.45);
            color: #FFFFFF;
        }}

        /* Sidebar Glass */
        [data-testid="stSidebar"] {{
            background: rgba(11, 17, 32, 0.96);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
        }}
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# BARRA LATERAL (SIDEBAR): PRESETS Y AMBIENTE
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="hero-badge">⚡ Panel Rápido</div>', unsafe_allow_html=True)
    st.markdown("### Perfiles Preconfigurados")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.86rem;'>"
        "Cargue perfiles típicos del sector comercial con <b>1 solo clic</b> para evaluar diversos escenarios:"
        "</p>",
        unsafe_allow_html=True
    )
    
    opciones_presets = list(PRESETS.keys())
    idx_actual = opciones_presets.index(st.session_state.preset_actual) if st.session_state.preset_actual in opciones_presets else 0
    
    st.selectbox(
        "Seleccionar Escenario / Plantilla:",
        options=opciones_presets,
        index=idx_actual,
        key="selector_preset",
        on_change=callback_cambio_preset,
        help="Al cambiar la opción, se poblarán instantáneamente todos los parámetros del asesor y tienda."
    )

    st.markdown("---")
    st.markdown("### 🎨 Entorno Visual")
    activar_fondo = st.toggle("Activar textura de fondo infográfica", value=False, help="Inyecta suavemente la imagen conceptual 'fondo_despliegue.jfif' como fondo con opacidad balanceada.")
    
    aplicar_estilos(activar_fondo_imagen=activar_fondo)

    st.markdown("---")
    st.markdown("### 🏬 Estacionalidad Retail")
    mes_actual = st.session_state.get("inp_mes_venta", "OCTUBRE")
    
    estacionalidad = {
        "DICIEMBRE": ("🔥 Temporada Muy Alta (Navidad & Fin de Año)", "#F43F5E"),
        "NOVIEMBRE": ("🛍️ Temporada Alta (Black Friday & Pre-Navidad)", "#FB923C"),
        "JUNIO": ("🏷️ Temporada Media-Alta (Primas Comerciales)", "#38BDF8"),
        "JULIO": ("🏷️ Temporada Media-Alta (Rebajas de Mitad de Año)", "#38BDF8"),
        "MAYO": ("💐 Temporada Media (Campaña Madres)", "#A855F7"),
        "ENERO": ("❄️ Temporada Baja (Post-Festividades)", "#94A3B8"),
        "FEBRERO": ("❄️ Temporada Baja", "#94A3B8")
    }
    
    desc_temp, color_temp = estacionalidad.get(mes_actual, ("📊 Temporada Regular", "#38BDF8"))
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); padding: 12px; border-radius: 10px; border-left: 4px solid {color_temp};">
            <div style="font-size:0.75rem; color:#94A3B8; font-weight:700;">CALENDARIO COMERCIAL ({mes_actual})</div>
            <div style="font-size:0.88rem; font-weight:700; color:{color_temp}; margin-top:3px;">{desc_temp}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ Especificaciones")
    st.caption("• **Modelo:** Gradient Boosting Regressor")
    st.caption("• **Precisión:** MAE estimado de ±3.3%")
    st.caption("• **Variables:** 38 features de entrada")
    st.caption("• **Target:** Facturación mensual en pesos COP")

# ==============================================================================
# ENCABEZADO PRINCIPAL (HERO)
# ==============================================================================
if error_carga:
    st.error(f"Error crítico al cargar 'modelo-gboosting-reg.pkl': {error_carga}")
    st.stop()

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-badge">✨ Retail Intelligence Platform • Fashion Stores</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-title">Proyección de Ventas por Asesor</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8; font-size:1.02rem; margin-top:2px; max-width:850px;'>"
        "Proyección analítica de la facturación comercial mensual del asesor comercial en el mes <b>T</b>, "
        "combinando el perfil del talento, contexto del punto de venta e historial trimestral."
        "</p>",
        unsafe_allow_html=True
    )

with col_h2:
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 14px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight:700;">Precisión Predictiva</div>
            <div style="font-size: 1.65rem; color: #38BDF8; font-weight: 800; font-family:'Space Grotesk';">96.7%</div>
            <div style="font-size: 0.75rem; color: #34D399; font-weight:600;">MAE Estimado ±3.3%</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# ==============================================================================
# PESTAÑAS PRINCIPALES
# ==============================================================================
tab_sim, tab_whatif, tab_meta, tab_metodologia = st.tabs([
    "🎯 Simulador & Proyección",
    "🔬 Simulador 'What-If' (Absentismo & Mes)",
    "🏆 Meta Comercial & Comisiones",
    "🧠 Arquitectura & Metodología"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: SIMULADOR & PROYECCIÓN PRINCIPAL
# ------------------------------------------------------------------------------
with tab_sim:
    st.markdown("### 📝 Parámetros Predictivos del Asesor y Tienda")
    
    col_asesor, col_tienda, col_historico = st.columns(3)

    # 1. PERFIL DEL ASESOR
    with col_asesor:
        st.markdown("""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:1.3rem;">👤</span>
                    <span style="font-weight:700; font-size:1.05rem; color:#FFFFFF;">Perfil del Asesor</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        edad_asesor = st.slider(
            'Edad del Asesor',
            min_value=18.0, max_value=60.0,
            step=1.0,
            key="inp_edad_asesor",
            help="Edad del asesor. Rango histórico del modelo: 14 a 57 años."
        )
        
        genero_asesor = st.selectbox(
            'Género',
            options=['F', 'M'],
            key="inp_genero_asesor",
            format_func=lambda x: "Femenino (F)" if x == "F" else "Masculino (M)"
        )
        
        nacionalidades = ['COLOMBIA', 'VENEZUELA', 'EXTRANJERO']
        nacionalidad_asesor = st.selectbox(
            'Nacionalidad',
            options=nacionalidades,
            key="inp_nacionalidad_asesor"
        )
        
        tipos_vinculacion = ['INDEF', 'OBRA']
        tipo_vinculacion = st.selectbox(
            'Tipo de Vinculación',
            options=tipos_vinculacion,
            key="inp_tipo_vinculacion",
            format_func=lambda x: "Término Indefinido (INDEF)" if x == "INDEF" else "Por Obra o Labor (OBRA)"
        )
        
        total_dias_absentismo = st.slider(
            'Días de Absentismo (Mes Actual)',
            min_value=0.0, max_value=30.0,
            step=1.0,
            key="inp_total_dias_absentismo",
            help="Días de ausencias, licencias médicas o permisos en el mes."
        )
        
        if total_dias_absentismo > 5:
            st.warning(f"⚠️ Absentismo relevante ({int(total_dias_absentismo)} días). Descuenta días efectivos de atención comercial.")

    # 2. CONTEXTO DE TIENDA
    with col_tienda:
        st.markdown("""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:1.3rem;">🏬</span>
                    <span style="font-weight:700; font-size:1.05rem; color:#FFFFFF;">Contexto de Tienda</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        marcas = ['MARCA_A', 'MARCA_B', 'MARCA_C', 'MARCA_D']
        marca = st.selectbox(
            'Línea / Marca de Moda',
            options=marcas,
            key="inp_marca"
        )
        
        zonas_comerciales = ['BOGOTA', 'MEDELLIN', 'CALI', 'BARRANQUILLA', 'CARTAGENA', 'EJE CAFETERO', 'SANTANDER', 'CENTRO', 'EXTERIOR']
        zona_comercial = st.selectbox(
            'Zona Comercial / Ciudad',
            options=zonas_comerciales,
            key="inp_zona_comercial"
        )
        
        ubicaciones = ['CENTRO COMERCIAL', 'EXTERIOR']
        tipo_ubicacion_tienda = st.selectbox(
            'Ubicación del Establecimiento',
            options=ubicaciones,
            key="inp_tipo_ubicacion_tienda",
            format_func=lambda x: "Centro Comercial (Alto Tráfico)" if x == "CENTRO COMERCIAL" else "Calle / Exterior"
        )
        
        cajas_col, m2_col = st.columns(2)
        with cajas_col:
            cantidad_cajas = st.number_input(
                'Puntos de Pago',
                min_value=1.0, max_value=4.0,
                step=1.0,
                key="inp_cantidad_cajas"
            )
        with m2_col:
            metros_cuadrados_tienda = st.number_input(
                'Área Local (m²)',
                min_value=50.0, max_value=330.0,
                step=10.0,
                key="inp_metros_cuadrados_tienda"
            )
            
        meses = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
        mes_venta = st.selectbox(
            'Mes a Proyectar (T)',
            options=meses,
            key="inp_mes_venta"
        )

    # 3. HISTORIAL DE VENTAS
    with col_historico:
        st.markdown("""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <span style="font-size:1.3rem;">📊</span>
                    <span style="font-weight:700; font-size:1.05rem; color:#FFFFFF;">Historial Trimestral (COP)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        venta_t_3 = st.number_input(
            'Venta Mes T-3 (Hace 3 meses)',
            min_value=0, max_value=500000000,
            step=500000,
            format="%d",
            key="inp_venta_t_3"
        )
        
        venta_t_2 = st.number_input(
            'Venta Mes T-2 (Hace 2 meses)',
            min_value=0, max_value=500000000,
            step=500000,
            format="%d",
            key="inp_venta_t_2"
        )
        
        venta_t_1 = st.number_input(
            'Venta Mes T-1 (Mes anterior)',
            min_value=0, max_value=500000000,
            step=500000,
            format="%d",
            key="inp_venta_t_1"
        )
        
        promedio_hist = (venta_t_3 + venta_t_2 + venta_t_1) / 3.0
        crecimiento_previo = ((venta_t_1 - venta_t_3) / venta_t_3 * 100) if venta_t_3 > 0 else 0.0
        
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:12px; margin-top:14px;">
                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Inercia Histórica</div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                    <span style="font-size:0.85rem; color:#CBD5E1;">Promedio T-3 a T-1:</span>
                    <span style="font-size:0.95rem; font-weight:700; color:#FFFFFF;">{formatear_cop(promedio_hist)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                    <span style="font-size:0.85rem; color:#CBD5E1;">Variación T-3 a T-1:</span>
                    <span style="font-size:0.95rem; font-weight:700; color:{'#34D399' if crecimiento_previo >= 0 else '#F87171'};">
                        {'+' if crecimiento_previo >= 0 else ''}{crecimiento_previo:.1f}%
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Diccionario compilado actual
    datos_actuales = {
        'edad_asesor': edad_asesor,
        'genero_asesor': genero_asesor,
        'nacionalidad_asesor': nacionalidad_asesor,
        'tipo_vinculacion': tipo_vinculacion,
        'total_dias_absentismo': total_dias_absentismo,
        'marca': marca,
        'zona_comercial': zona_comercial,
        'tipo_ubicacion_tienda': tipo_ubicacion_tienda,
        'cantidad_cajas': cantidad_cajas,
        'metros_cuadrados_tienda': metros_cuadrados_tienda,
        'mes_venta': mes_venta,
        'venta_t_1': venta_t_1,
        'venta_t_2': venta_t_2,
        'venta_t_3': venta_t_3
    }

    st.write("")
    ejecutar = st.button("🔮 GENERAR PROYECCIÓN COMERCIAL", use_container_width=True)

    if ejecutar or 'ultima_prediccion' in st.session_state:
        if ejecutar or 'ultima_prediccion' not in st.session_state:
            with st.spinner("Computando inferencia sobre Gradient Boosting Regressor..."):
                time.sleep(0.3)
                pred_cop = predecir_venta(datos_actuales)
                st.session_state.ultima_prediccion = pred_cop
        else:
            pred_cop = st.session_state.ultima_prediccion

        st.markdown("---")
        st.markdown("## 📈 Resultados Ejecutivos del Pronóstico")

        dif_t1 = pred_cop - venta_t_1
        pct_t1 = (dif_t1 / venta_t_1 * 100) if venta_t_1 > 0 else 0.0
        
        dif_prom = pred_cop - promedio_hist
        pct_prom = (dif_prom / promedio_hist * 100) if promedio_hist > 0 else 0.0
        
        dias_laborales_efectivos = max(1.0, 30.0 - total_dias_absentismo)
        venta_diaria = pred_cop / dias_laborales_efectivos
        venta_por_m2 = pred_cop / max(1.0, metros_cuadrados_tienda)

        # KPIs EN COLUMNAS
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Venta Proyectada (Mes {mes_venta})</div>
                    <div class="kpi-value" style="color:#38BDF8;">{formatear_cop(pred_cop)}</div>
                    <div class="kpi-sub">Pronóstico mensual estimado</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            clase_pill_t1 = "pill-positive" if dif_t1 >= 0 else "pill-negative"
            signo_t1 = "+" if dif_t1 >= 0 else ""
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Crecimiento vs Mes T-1</div>
                    <div class="kpi-value">{signo_t1}{pct_t1:.1f}%</div>
                    <div class="kpi-sub">
                        <span class="{clase_pill_t1}">{signo_t1}{formatear_cop(dif_t1)}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with kpi3:
            clase_pill_prom = "pill-positive" if dif_prom >= 0 else "pill-negative"
            signo_prom = "+" if dif_prom >= 0 else ""
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">vs Promedio Trimestral</div>
                    <div class="kpi-value">{signo_prom}{pct_prom:.1f}%</div>
                    <div class="kpi-sub">
                        <span class="{clase_pill_prom}">{signo_prom}{formatear_cop(dif_prom)}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Venta Diaria Promedio</div>
                    <div class="kpi-value" style="font-size:1.85rem;">{formatear_cop(venta_diaria)}</div>
                    <div class="kpi-sub">{int(dias_laborales_efectivos)} días activos proyectados</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")

        # GRÁFICA INTERACTIVA Y PANEL DE DIAGNÓSTICO
        col_grafica, col_diagnostico = st.columns([1.6, 1.0])

        with col_grafica:
            st.markdown("#### 📊 Evolución Histórica y Pronóstico del Asesor")
            
            df_chart = pd.DataFrame({
                "Periodo": ["Mes T-3", "Mes T-2", "Mes T-1", f"Mes T ({mes_venta})*"],
                "Tipo": ["Histórico", "Histórico", "Histórico", "Proyección"],
                "Ventas": [venta_t_3, venta_t_2, venta_t_1, pred_cop],
                "Orden": [1, 2, 3, 4]
            })

            base = alt.Chart(df_chart).encode(
                x=alt.X('Periodo:N', sort=alt.EncodingSortField(field="Orden", order="ascending"), title=None, axis=alt.Axis(labelColor="#CBD5E1", labelFontSize=12)),
                y=alt.Y('Ventas:Q', title="Facturación en Pesos (COP)", axis=alt.Axis(format="$,.0f", labelColor="#CBD5E1", titleColor="#94A3B8"))
            )

            area = base.mark_area(
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color='rgba(56, 189, 248, 0.4)', offset=0),
                        alt.GradientStop(color='rgba(56, 189, 248, 0.02)', offset=1)
                    ],
                    x1=1, x2=1, y1=1, y2=0
                )
            )

            line = base.mark_line(color="#38BDF8", strokeWidth=3)

            points = base.mark_circle(size=120).encode(
                color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Histórico', 'Proyección'], range=['#38BDF8', '#F59E0B']), legend=alt.Legend(title="Naturaleza", orient="bottom", labelColor="#CBD5E1")),
                tooltip=[
                    alt.Tooltip('Periodo:N', title='Periodo'),
                    alt.Tooltip('Tipo:N', title='Naturaleza'),
                    alt.Tooltip('Ventas:Q', title='Facturación (COP)', format="$,.0f")
                ]
            )

            chart_final = (area + line + points).properties(height=320)
            st.altair_chart(chart_final, use_container_width=True)

        with col_diagnostico:
            st.markdown("#### 🎯 Diagnóstico Estratégico")
            
            if pct_t1 >= 10.0:
                color_diag = "#34D399"
                estado_diag = "🌟 Crecimiento Acelerado (Sobresaliente)"
                consejo = "El colaborador muestra una marcada curva ascendente. Se sugiere elevar la meta con bonos escalonados y asegurar disponibilidad de tallas y productos top en su estación."
            elif pct_t1 >= 0.0:
                color_diag = "#38BDF8"
                estado_diag = "✅ Rendimiento Sólido & Estable"
                consejo = "El colaborador mantiene un desempeño constante. Mantener seguimiento y aplicar incentivos en venta cruzada (cross-selling) para incrementar el ticket promedio."
            elif pct_t1 >= -10.0:
                color_diag = "#FBBF24"
                estado_diag = "⚠️ Desaceleración Moderada"
                consejo = "Se proyecta una leve contracción. Corroborar si coincide con tráfico bajo del mes o días de ausencia. Reforzar acompañamiento en técnicas de cierre de venta."
            else:
                color_diag = "#F87171"
                estado_diag = "🚨 Alerta de Descenso Comercial"
                consejo = "Descenso proyectado significativo. Se recomienda revisar el impacto de ausentismo o evaluar rotación de turnos a horarios pico de afluencia."

            st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid {color_diag}; padding: 18px;">
                    <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Estado Predictivo</div>
                    <div style="font-size:1.1rem; font-weight:700; color:{color_diag}; margin-top:4px;">{estado_diag}</div>
                    <div style="font-size:0.88rem; color:#CBD5E1; margin-top:12px; line-height:1.5;">
                        <b>Sugerencia Gerencial:</b><br>{consejo}
                    </div>
                    <div style="margin-top:14px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08); font-size:0.8rem; color:#94A3B8;">
                        • Rendimiento por m²: <b style="color:#FFFFFF;">{formatear_cop(venta_por_m2)} / m²</b><br>
                        • Ubicación: <b style="color:#FFFFFF;">{tipo_ubicacion_tienda}</b> en {zona_comercial}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # DESCARGA DE REPORTE CSV
        st.write("")
        df_export = pd.DataFrame([{
            "fecha_reporte": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mes_proyectado": mes_venta,
            "venta_proyectada_cop": round(pred_cop, 2),
            "edad_asesor": edad_asesor,
            "genero": genero_asesor,
            "nacionalidad": nacionalidad_asesor,
            "tipo_vinculacion": tipo_vinculacion,
            "dias_absentismo": total_dias_absentismo,
            "marca": marca,
            "zona_comercial": zona_comercial,
            "tipo_ubicacion": tipo_ubicacion_tienda,
            "cantidad_cajas": cantidad_cajas,
            "m2_tienda": metros_cuadrados_tienda,
            "venta_t_1": venta_t_1,
            "venta_t_2": venta_t_2,
            "venta_t_3": venta_t_3,
            "variacion_pct_vs_t1": round(pct_t1, 2)
        }])
        
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Ficha de Proyección (.CSV)",
            data=csv_data,
            file_name=f"proyeccion_asesor_{mes_venta}_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Descargue un informe completo para su integración con planes comerciales o ERP."
        )

# ------------------------------------------------------------------------------
# PESTAÑA 2: SIMULADOR 'WHAT-IF'
# ------------------------------------------------------------------------------
with tab_whatif:
    st.markdown("### 🔬 Laboratorio 'What-If': Sensibilidad y Estacionalidad")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.92rem;'>"
        "Explore el impacto directo que causan el <b>absentismo laboral</b> y los <b>12 meses del año</b> "
        "sobre la capacidad facturadora del asesor."
        "</p>",
        unsafe_allow_html=True
    )
    
    col_w1, col_w2 = st.columns(2)
    
    # 1. IMPACTO DEL ABSENTISMO
    with col_w1:
        st.markdown("""
            <div class="glass-card">
                <div style="font-weight:700; font-size:1.05rem; color:#FFFFFF; margin-bottom:6px;">
                    🛡️ Sensibilidad por Días de Absentismo
                </div>
                <p style="color:#94A3B8; font-size:0.85rem;">
                    Compare la venta esperada si el asesor estuviera al 100% (0 días) versus diferentes niveles de ausentismo.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        sim_dias_abs = st.slider("Simular Días de Ausencia:", min_value=0.0, max_value=25.0, value=float(st.session_state.inp_total_dias_absentismo), step=1.0)
        
        d_cero = datos_actuales.copy()
        d_cero['total_dias_absentismo'] = 0.0
        v_optima = predecir_venta(d_cero)
        
        d_sim = datos_actuales.copy()
        d_sim['total_dias_absentismo'] = sim_dias_abs
        v_simulada = predecir_venta(d_sim)
        
        costo_oportunidad = v_optima - v_simulada
        
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Venta Proyectada Simulada", formatear_cop(v_simulada))
        with m_c2:
            st.metric("Pérdida por Absentismo", f"- {formatear_cop(costo_oportunidad)}", delta_color="inverse")
            
        lista_dias = [0.0, 3.0, 7.0, 10.0, 15.0, 20.0]
        ventas_sens = []
        for d in lista_dias:
            dt = datos_actuales.copy()
            dt['total_dias_absentismo'] = d
            ventas_sens.append(predecir_venta(dt))
            
        df_sens = pd.DataFrame({
            "Días de Absentismo": [f"{int(d)} d" for d in lista_dias],
            "Orden": lista_dias,
            "Venta Estimada": ventas_sens
        })
        
        chart_sens = alt.Chart(df_sens).mark_bar(color="#F43F5E", cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X('Días de Absentismo:N', sort=alt.EncodingSortField(field="Orden", order="ascending"), title="Días Ausente", axis=alt.Axis(labelColor="#CBD5E1")),
            y=alt.Y('Venta Estimada:Q', title="Venta COP", axis=alt.Axis(format="$,.0f", labelColor="#CBD5E1")),
            tooltip=[alt.Tooltip('Días de Absentismo:N'), alt.Tooltip('Venta Estimada:Q', format="$,.0f")]
        ).properties(height=240)
        
        st.altair_chart(chart_sens, use_container_width=True)

    # 2. CURVA ANUAL (12 MESES)
    with col_w2:
        st.markdown("""
            <div class="glass-card">
                <div style="font-weight:700; font-size:1.05rem; color:#FFFFFF; margin-bottom:6px;">
                    📅 Curva de Rendimiento en los 12 Meses
                </div>
                <p style="color:#94A3B8; font-size:0.85rem;">
                    Proyección del asesor a lo largo de todo el año bajo las mismas características de tienda e historial.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        meses_todos = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
        
        with st.spinner("Generando proyección para los 12 meses..."):
            ventas_meses = []
            for m in meses_todos:
                dm = datos_actuales.copy()
                dm['mes_venta'] = m
                ventas_meses.append(predecir_venta(dm))
                
        df_meses = pd.DataFrame({
            "Mes": [m[:3] for m in meses_todos],
            "Mes_Completo": meses_todos,
            "Orden": range(1, 13),
            "Venta": ventas_meses
        })
        
        mes_seleccionado = datos_actuales.get('mes_venta', 'OCTUBRE')
        
        chart_meses = alt.Chart(df_meses).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X('Mes:N', sort=alt.EncodingSortField(field="Orden", order="ascending"), title=None, axis=alt.Axis(labelColor="#CBD5E1")),
            y=alt.Y('Venta:Q', title="Venta Estimada (COP)", axis=alt.Axis(format="$,.0f", labelColor="#CBD5E1")),
            color=alt.condition(
                alt.datum.Mes_Completo == mes_seleccionado,
                alt.value('#38BDF8'),
                alt.value('#334155')
            ),
            tooltip=[alt.Tooltip('Mes_Completo:N', title='Mes'), alt.Tooltip('Venta:Q', format="$,.0f", title='Venta Proyectada')]
        ).properties(height=280)
        
        st.altair_chart(chart_meses, use_container_width=True)
        st.caption(f"💡 La barra resaltada en celeste corresponde al mes seleccionado: **{mes_seleccionado}**.")

# ------------------------------------------------------------------------------
# PESTAÑA 3: METAS Y COMISIONES
# ------------------------------------------------------------------------------
with tab_meta:
    st.markdown("### 🏆 Calculadora de Cumplimiento de Metas & Comisiones")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.92rem;'>"
        "Valide la viabilidad de la cuota comercial asignada al asesor y calcule las comisiones y bonos proyectados."
        "</p>",
        unsafe_allow_html=True
    )
    
    col_m1, col_m2 = st.columns(2)
    v_actual = st.session_state.get('ultima_prediccion', predecir_venta(datos_actuales))
    
    with col_m1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Parámetros de Cuota Comercial")
        meta_cuota = st.number_input(
            "Meta Asignada al Asesor (COP)",
            min_value=1000000,
            max_value=200000000,
            value=int(v_actual * 0.95),
            step=500000,
            format="%d",
            help="Cuota mensual de ventas fijada al asesor."
        )
        
        pct_comision_base = st.slider(
            "Comisión Base sobre Ventas",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            format="%.1f%%"
        )
        
        bono_cumplimiento = st.number_input(
            "Bono por Cumplimiento de Meta (COP)",
            min_value=0,
            max_value=5000000,
            value=300000,
            step=50000,
            format="%d"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        cumplimiento_pct = (v_actual / meta_cuota * 100) if meta_cuota > 0 else 0.0
        comision_estimada = (v_actual * (pct_comision_base / 100.0))
        if cumplimiento_pct >= 100.0:
            comision_estimada += bono_cumplimiento
            estado_meta = "🏆 META CUMPLIDA (Accede a Bono)"
            color_meta = "#34D399"
        else:
            estado_meta = "⏳ EN PROCESO DE CUMPLIMIENTO"
            color_meta = "#FBBF24"

        st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid {color_meta};">
                <div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Cumplimiento Estimado de Cuota</div>
                <div style="font-size:2.4rem; font-weight:800; color:{color_meta}; font-family:'Space Grotesk';">
                    {cumplimiento_pct:.1f}%
                </div>
                <div style="font-size:0.95rem; font-weight:600; color:#FFFFFF; margin-top:2px;">{estado_meta}</div>
                <div style="margin-top:16px; padding:12px; background:rgba(0,0,0,0.25); border-radius:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="color:#CBD5E1;">Venta Proyectada:</span>
                        <b style="color:#FFFFFF;">{formatear_cop(v_actual)}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="color:#CBD5E1;">Meta Asignada:</span>
                        <b style="color:#FFFFFF;">{formatear_cop(meta_cuota)}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding-top:6px; border-top:1px solid rgba(255,255,255,0.08);">
                        <span style="color:#38BDF8; font-weight:600;">Comisión Estimada Total:</span>
                        <b style="color:#38BDF8; font-size:1.1rem;">{formatear_cop(comision_estimada)}</b>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        progreso = min(1.0, max(0.0, float(cumplimiento_pct / 100.0)))
        st.progress(progreso)

# ------------------------------------------------------------------------------
# PESTAÑA 4: METODOLOGÍA & ARQUITECTURA
# ------------------------------------------------------------------------------
with tab_metodologia:
    st.markdown("### 🧠 Arquitectura del Modelo Predictivo")
    st.markdown(
        "<p style='color:#94A3B8; font-size:0.92rem;'>"
        "Infografía y especificaciones del algoritmo Gradient Boosting Regressor entrenado para el retail de moda."
        "</p>",
        unsafe_allow_html=True
    )
    
    try:
        st.image("fondo_despliegue.jfif", caption="Flujo de Inferencia: Perfil Asesor + Contexto Tienda + Historial Ventas ➔ Gradient Boosting Model ➔ Venta Proyectada Mes T", use_container_width=True)
    except Exception:
        st.info("Visualización infográfica disponible en el repositorio.")

    st.markdown("---")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color:#38BDF8; margin-bottom:8px;">📌 Variables Predictoras</h4>
                <ul style="color:#CBD5E1; font-size:0.9rem; line-height:1.7;">
                    <li><b>Perfil del Asesor:</b> Edad, género, nacionalidad, tipo de contrato y absentismo del mes proyectado.</li>
                    <li><b>Contexto del Punto de Venta:</b> Marca, zona geográfica (9 regiones), ubicación (centro comercial o exterior), cajas de cobro y superficie en m².</li>
                    <li><b>Inercia Histórica:</b> Facturación real de los meses T-1, T-2 y T-3 para capturar tendencias del asesor.</li>
                    <li><b>Calendario Comercial:</b> Mes a proyectar para incorporar la estacionalidad (Navidad, día de la madre, etc.).</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("""
            <div class="glass-card">
                <h4 style="color:#34D399; margin-bottom:8px;">⚙️ Pipeline de Transformación</h4>
                <ul style="color:#CBD5E1; font-size:0.9rem; line-height:1.7;">
                    <li><b>One-Hot Encoding:</b> Binarización de categorías sin pérdida de dimensionalidad (38 features resultantes).</li>
                    <li><b>Normalización MinMaxScaler:</b> Escalado acotado [0, 1] de variables numéricas para balance numérico.</li>
                    <li><b>Algoritmo Regresor:</b> Gradient Boosting Regressor (ensamble secuencial de árboles de decisión).</li>
                    <li><b>Transformación Inversa:</b> Reescalado exacto a pesos colombianos (COP).</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PIE DE PÁGINA (FOOTER)
# ==============================================================================
st.write("")
st.markdown("---")
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    st.markdown(
        "<div style='color:#64748B; font-size:0.82rem;'>"
        "© Fashion Retail Intelligence • Modelo de Regresión para Proyección Comercial de Asesores • Todos los derechos reservados."
        "</div>",
        unsafe_allow_html=True
    )
with col_f2:
    st.markdown(
        "<div style='text-align:right; color:#64748B; font-size:0.82rem;'>"
        "Desplegado con <b>Streamlit</b> & <b>Scikit-Learn 1.6.1</b>"
        "</div>",
        unsafe_allow_html=True
    )
