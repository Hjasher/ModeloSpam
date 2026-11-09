import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


NOMBRE = 'Vianey Diaz - Hjasher Medina'
TITULO = 'Detector de spam con SVM y árbol de decisión'

CARPETA = Path(__file__).resolve().parent
RUTA_BD = CARPETA / 'spam.db'

st.set_page_config(page_title='Detector de spam', layout='wide')
st.title(TITULO)
st.markdown(f'**Estudiante** {NOMBRE}')
st.caption('Inteligencia Artificial | Actividad 4 grupal')


@st.cache_resource
def cargar_modelos():
    svm = joblib.load(CARPETA / 'svm_spam.pkl')     
    arbol = joblib.load(CARPETA / 'arbol_spam.pkl')
    with open(CARPETA / 'metricas.json', encoding='utf-8') as f:
        metricas = json.load(f)
    return svm, arbol, metricas


svm, arbol, metricas = cargar_modelos()
CLASES = metricas['clases']                    # ['Normal', 'Spam']
CARACTERISTICAS = metricas['caracteristicas']  # ['enlaces', 'palabras_promocionales']



def ejecutar(sql, parametros=()):
    with closing(sqlite3.connect(RUTA_BD)) as con:
        with con:
            con.execute(sql, parametros)


def crear_tabla():
    ejecutar('''
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT NOT NULL,
            enlaces INTEGER NOT NULL,
            palabras_promocionales INTEGER NOT NULL,
            prediccion_svm TEXT NOT NULL,
            prediccion_arbol TEXT NOT NULL
        )
    ''')


def registrar_consulta(enlaces, palabras, pred_svm, pred_arbol):
    ejecutar(
        '''INSERT INTO consultas
           (fecha_hora, enlaces, palabras_promocionales, prediccion_svm, prediccion_arbol)
           VALUES (?, ?, ?, ?, ?)''',
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), enlaces, palabras, pred_svm, pred_arbol),
    )


def leer_historial():
    with closing(sqlite3.connect(RUTA_BD)) as con:
        return pd.read_sql_query(
            '''SELECT id AS "N°",
                      fecha_hora AS "Fecha y hora",
                      enlaces AS "Enlaces",
                      palabras_promocionales AS "Palabras promocionales",
                      prediccion_svm AS "Predicción SVM",
                      prediccion_arbol AS "Predicción árbol"
               FROM consultas ORDER BY id DESC''',
            con,
        )


crear_tabla()

st.header('1. Clasificar un mensaje')

col1, col2 = st.columns(2)
enlaces = col1.number_input('Cantidad de enlaces', min_value=0, max_value=100, value=0, step=1)
palabras = col2.number_input('Cantidad de palabras promocionales', min_value=0, max_value=100, value=0, step=1)

if st.button('Clasificar', type='primary'):
    entrada = pd.DataFrame([{
        CARACTERISTICAS[0]: int(enlaces),
        CARACTERISTICAS[1]: int(palabras),
    }])
    pred_svm = CLASES[int(svm.predict(entrada)[0])]
    pred_arbol = CLASES[int(arbol.predict(entrada)[0])]

    registrar_consulta(int(enlaces), int(palabras), pred_svm, pred_arbol)

    st.session_state['ultima'] = {
        'enlaces': int(enlaces),
        'palabras': int(palabras),
        'svm': pred_svm,
        'arbol': pred_arbol,
    }

ultima = st.session_state.get('ultima')
if ultima:
    st.markdown(
        f"Resultado para **{ultima['enlaces']}** enlaces y "
        f"**{ultima['palabras']}** palabras promocionales"
    )
    r1, r2 = st.columns(2)
    r1.metric('SVM', ultima['svm'])
    r2.metric('Árbol de decisión', ultima['arbol'])

    if ultima['svm'] == ultima['arbol']:
        st.success(f"Los dos modelos coinciden, ambos dicen {ultima['svm']}.")
    else:
        st.warning(
            f"Los modelos no coinciden, la SVM dice {ultima['svm']} "
            f"y el árbol dice {ultima['arbol']}."
        )
else:
    st.info('Ingrese los valores y presione «Clasificar».')

with st.expander('Cómo se calcularon las dos características'):
    st.write(
        'La base de datos (SMS Spam Collection) solo trae el texto de cada mensaje y su etiqueta, '
        'por eso las dos características numéricas se calcularon a partir del texto.'
    )
    st.write(
        '**Enlaces** son las direcciones tipo http://, www. o dominios terminados en .com, .net, .org y similares.'
    )
    st.write(
        '**Palabras promocionales** son las que aparecen en una lista fija de '
        f"{len(metricas['palabras_promocionales'])} palabras, definida antes de ver las etiquetas."
    )
    st.code(', '.join(metricas['palabras_promocionales']), language=None)

st.divider()


# Comparación de los modelos

st.header('2. Comparación de exactitud')

nombres_modelos = list(metricas['modelos'].keys())
exactitudes = [metricas['modelos'][n]['exactitud'] for n in nombres_modelos]
referencia = metricas['exactitud_referencia_todo_normal']

g1, g2 = st.columns([1, 2])

with g1:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    barras = ax.bar(nombres_modelos, exactitudes, color=['#4C78A8', '#F58518'])
    ax.axhline(referencia, color='gray', linestyle='--', label=f'Decir siempre «Normal» ({referencia:.4f})')
    for barra, valor in zip(barras, exactitudes):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.01, f'{valor:.4f}', ha='center')
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Exactitud en el conjunto de prueba')
    ax.set_title('Exactitud de cada modelo')
    ax.legend(loc='upper center', fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

with g2:
    fig, ejes = plt.subplots(1, 2, figsize=(8, 4))
    for eje, nombre in zip(ejes, nombres_modelos):
        matriz = np.array(metricas['modelos'][nombre]['matriz_confusion'])
        ConfusionMatrixDisplay(matriz, display_labels=CLASES).plot(ax=eje, colorbar=False)
        eje.set_title(f'Matriz de confusión\n{nombre}')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

tabla = pd.DataFrame({
    nombre: {
        'Exactitud': m['exactitud'],
        'Precisión (spam)': m['precision_spam'],
        'Sensibilidad (spam)': m['recall_spam'],
        'F1 (spam)': m['f1_spam'],
    }
    for nombre, m in metricas['modelos'].items()
}).T
st.dataframe(tabla.style.format('{:.4f}'))
st.caption(
    f"Conjunto de prueba de {metricas['n_prueba']} mensajes. "
    f"Los dos modelos coinciden en {metricas['coinciden_en_prueba']} de ellos."
)

st.divider()


# Árbol de decisión

st.header('3. Árbol de decisión')

fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    arbol,
    feature_names=CARACTERISTICAS,
    class_names=CLASES,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
)
st.pyplot(fig)
plt.close(fig)

st.divider()


# Historial

st.header('4. Historial de consultas')

historial = leer_historial()
st.write(f'Consultas registradas en spam.db {len(historial)}')
if historial.empty:
    st.info('Todavía no hay consultas guardadas.')
else:
    st.dataframe(historial, hide_index=True)
st.caption('Las consultas se guardan en el archivo spam.db, por eso permanecen al reiniciar la aplicación.')
