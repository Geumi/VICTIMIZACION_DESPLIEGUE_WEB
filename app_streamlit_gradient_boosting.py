# ============================================================
# APP STREAMLIT - MODELO DE VICTIMIZACIÓN
# GRADIENT BOOSTING
# ============================================================

import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path


# ============================================================
# 01. CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Modelo predictivo de victimización",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# 02. CARGAR MODELO
# ============================================================

# Se asume que el .joblib se encuentra en la misma carpeta
# que este archivo app_streamlit.py.
RUTA_APP = Path(__file__).resolve().parent
RUTA_MODELO = RUTA_APP / "modelo_gradient_boosting_victimizacion.joblib"

try:
    modelo_guardado = load(RUTA_MODELO)
except FileNotFoundError:
    st.error(
        "No se encontró el archivo "
        "'modelo_gradient_boosting_victimizacion.joblib' "
        "en la misma carpeta que app_streamlit.py."
    )
    st.stop()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()


# El archivo .joblib nuevo guarda un diccionario.
if not isinstance(modelo_guardado, dict):
    st.error(
        "El archivo cargado no tiene la estructura esperada. "
        "Vuelve a ejecutar el entrenamiento y guarda el modelo "
        "con el código actualizado."
    )
    st.stop()


claves_necesarias = {
    "modelo",
    "umbral",
    "variables_categoricas",
    "variables_numericas",
    "variables_modelo"
}

faltantes = claves_necesarias.difference(modelo_guardado.keys())

if faltantes:
    st.error(
        "Al archivo del modelo le faltan estas claves: "
        + ", ".join(sorted(faltantes))
    )
    st.stop()


# Pipeline completo entrenado:
# preprocesamiento + Gradient Boosting
clf = modelo_guardado["modelo"]

# Umbral elegido durante el entrenamiento
umbral = float(modelo_guardado["umbral"])

# Variables guardadas con el modelo
columnas_categoricas = list(
    modelo_guardado["variables_categoricas"]
)

columnas_numericas = list(
    modelo_guardado["variables_numericas"]
)

variables_modelo = list(
    modelo_guardado["variables_modelo"]
)


# ============================================================
# 03. RECUPERAR CATEGORÍAS DESDE EL PIPELINE
# ============================================================

try:
    preproc = clf.named_steps["preproc"]

    # En el modelo nuevo el transformador categórico se llama "cat"
    # y dentro tiene:
    # SimpleImputer + OneHotEncoder
    pipeline_cat = preproc.named_transformers_["cat"]

    encoder = pipeline_cat.named_steps["encoder"]

except Exception as e:
    st.error(
        "No se pudo recuperar el preprocesador categórico "
        f"desde el modelo: {e}"
    )
    st.stop()


# El orden de encoder.categories_ corresponde al orden en que
# se entrenaron las variables categóricas.
opciones = {
    columna: categorias.tolist()
    for columna, categorias in zip(
        columnas_categoricas,
        encoder.categories_
    )
}


# ============================================================
# 04. VALIDAR VARIABLES ESPERADAS
# ============================================================

variables_esperadas = {
    "P207",        # Sexo
    "P208_A",      # Edad
    "P308",        # Nivel educativo
    "VIGILANCIA",  # Vigilancia
    "ESTRATO",     # Estrato
    "NOMBREDD"     # Departamento
}

variables_faltantes = (
    variables_esperadas.difference(variables_modelo)
)

if variables_faltantes:
    st.error(
        "El modelo cargado no contiene todas las variables "
        "esperadas por esta app. Faltan: "
        + ", ".join(sorted(variables_faltantes))
    )
    st.stop()


# Recuperar opciones directamente del modelo
sexo_options = opciones["P207"]
p308_options = opciones["P308"]
vigilancia_options = opciones["VIGILANCIA"]
estrato_options = opciones["ESTRATO"]
departamento_options = opciones["NOMBREDD"]


# ============================================================
# 05. RECUPERAR CLASE POSITIVA
# ============================================================

try:
    modelo_gb = clf.named_steps["gb"]
    clases = list(modelo_gb.classes_)
    indice_victima = clases.index(1)
except Exception as e:
    st.error(
        "No se pudo identificar la clase Víctima = 1 "
        f"en el modelo: {e}"
    )
    st.stop()


# ============================================================
# 06. FUNCIONES AUXILIARES
# ============================================================

def reset_inputs():
    """
    Elimina el estado de los widgets y vuelve a cargar la app.
    """
    claves_widgets = [
        "sexo_input",
        "edad_input",
        "estrato_input",
        "educacion_input",
        "vigilancia_input",
        "departamento_input"
    ]

    for clave in claves_widgets:
        if clave in st.session_state:
            del st.session_state[clave]

    st.rerun()


# ============================================================
# 07. TÍTULO
# ============================================================

st.title("Modelo predictivo de victimización")

st.markdown(
    """
    Estimación exploratoria de la probabilidad de victimización
    mediante un modelo **Gradient Boosting**.

    El modelo utiliza sexo, edad, nivel educativo, vigilancia,
    estrato y departamento.
    """
)

st.markdown("**Elaborado por Sandra Nuñez y David Cruz**")


st.markdown("---")


# ============================================================
# 08. FORMULARIO
# ============================================================

with st.form("formulario_victimizacion"):

    col1, col2 = st.columns(2)

    with col1:

        P207 = st.selectbox(
            "**SEXO**",
            sexo_options,
            key="sexo_input"
        )

        P208_A = st.number_input(
            "**EDAD**",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
            key="edad_input"
        )

        ESTRATO = st.selectbox(
            "**ESTRATO**",
            estrato_options,
            key="estrato_input"
        )

    with col2:

        P308 = st.selectbox(
            "**NIVEL EDUCATIVO**",
            p308_options,
            key="educacion_input"
        )

        VIGILANCIA = st.selectbox(
            "**VIGILANCIA EN LA ZONA**",
            vigilancia_options,
            key="vigilancia_input"
        )

        NOMBREDD = st.selectbox(
            "**DEPARTAMENTO**",
            departamento_options,
            key="departamento_input"
        )

    predict_button = st.form_submit_button(
        "Predecir",
        use_container_width=True
    )


# ============================================================
# 09. PREDICCIÓN
# ============================================================

if predict_button:

    # El DataFrame debe tener exactamente las mismas variables
    # usadas durante el entrenamiento.
    nuevo_dato = pd.DataFrame(
        {
            "P207": [str(P207)],
            "P208_A": [float(P208_A)],
            "P308": [str(P308)],
            "VIGILANCIA": [str(VIGILANCIA)],
            "ESTRATO": [str(ESTRATO)],
            "NOMBREDD": [str(NOMBREDD)]
        }
    )

    # Reordenar usando exactamente el orden guardado
    nuevo_dato = nuevo_dato[variables_modelo]

    try:
        probabilidades = clf.predict_proba(nuevo_dato)[0]
        prob_victima = float(
            probabilidades[indice_victima]
        )

    except Exception as e:
        st.error(
            "Se produjo un error al realizar la predicción: "
            f"{e}"
        )
        st.stop()


    # --------------------------------------------------------
    # Clasificación utilizando el umbral ajustado
    # NO usamos clf.predict(), porque usaría el criterio
    # estándar asociado al umbral 0.50.
    # --------------------------------------------------------

    es_victima = prob_victima >= umbral


    # ========================================================
    # 10. MOSTRAR RESULTADO
    # ========================================================

    st.markdown("---")
    st.subheader("Resultado")

    st.metric(
        label="Probabilidad estimada de victimización",
        value=f"{prob_victima:.1%}"
    )

    if es_victima:

        st.warning(
            "La probabilidad estimada supera el umbral "
            f"de clasificación del modelo ({umbral:.1%})."
        )

        st.markdown(
            "**Clasificación del modelo:** "
            "Mayor riesgo de victimización."
        )

    else:

        st.success(
            "La probabilidad estimada no supera el umbral "
            f"de clasificación del modelo ({umbral:.1%})."
        )

        st.markdown(
            "**Clasificación del modelo:** "
            "Menor riesgo de victimización."
        )


    # Mostrar información adicional
    with st.expander("Ver detalle técnico"):

        st.write(
            "Probabilidad de Víctima:",
            round(prob_victima, 4)
        )

        st.write(
            "Probabilidad de No víctima:",
            round(1 - prob_victima, 4)
        )

        st.write(
            "Umbral utilizado:",
            round(umbral, 4)
        )

        st.write(
            "Variables ingresadas:"
        )

        st.dataframe(
            nuevo_dato,
            use_container_width=True
        )


# ============================================================
# 11. BOTÓN RESET
# ============================================================

if st.button(
    "Resetear",
    use_container_width=True
):
    reset_inputs()


# ============================================================
# PARA EJECUTAR EN LA TERMINAL:
#
# streamlit run app_streamlit_gradient_boosting.py
#
# Si usas un entorno virtual:
#
# .\.venv\Scripts\activate
# streamlit run app_streamlit.py
#
# Dependencias principales:
#
# pip install streamlit pandas joblib scikit-learn
# ============================================================
