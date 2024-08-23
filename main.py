from langchain.embeddings import OpenAIEmbeddings    
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.prompts import ChatPromptTemplate
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Union
import math
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql, Error

host = 'db-resumescreening-coally.c960wcwwcazt.us-east-2.rds.amazonaws.com'
dbname = 'postgres'
user = 'postgres'
password = 'CoallySecur3'

class FiltroRequest(BaseModel):
    role: str | None = None
    ranking_maximo: int | None = None
    origen: str | None = None
    precio_maximo: str | None = None
    metodologia:str | None = None

load_dotenv()

class MajorRecommender:

    def __init__(self):
        self.qa_stuff = None
        self.llm = None
        self.df = pd.read_csv('ProgramasCompletos.csv')


    def prepare_qa_chain(self, filtro):

        loader = CSVLoader("ProgramasCompletos.csv", encoding='utf-8', 
                           metadata_columns=['Ranking institución educativa', 'Origen institución educativa', 'Metodología programa educativo', 'precio'])
        docs = loader.load()
        self.llm = ChatOpenAI(temperature = 0.0, openai_api_key= os.environ.get('OPENAI_API_KEY'), model ='gpt-4o-mini')

        embeddings = OpenAIEmbeddings(openai_api_key= os.environ.get('OPENAI_API_KEY'))
        filtered_docs = []


        for doc in docs:
            if ('ranking_maximo' in filtro and filtro['ranking_maximo'] is not None and filtro['ranking_maximo'] != 0 and int(doc.metadata.get('Ranking institución educativa')) >= filtro['ranking_maximo']):
                continue
            if ('origen' in filtro and filtro['origen'] != "" and doc.metadata.get('Origen institución educativa') != filtro['origen']):
                continue
            if ('precio_maximo' in filtro and filtro['precio_maximo'] != "" and len(doc.metadata.get('precio')) > len(filtro['precio_maximo'])):
                continue
            if ('metodologia' in filtro and filtro['metodologia'] != "" and doc.metadata.get('Metodología programa educativo') != filtro['metodologia']):
                continue

            filtered_docs.append(doc)

        db = DocArrayInMemorySearch.from_documents(filtered_docs, embeddings)

        retriever = db.as_retriever(search_kwargs={"k": 15})

        qa_stuff = RetrievalQA.from_chain_type(
        llm=self.llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
        )

        return qa_stuff

    def get_chain_response(self, role, filtro = {}):

        qa_stuff = self.prepare_qa_chain(filtro)

        query =  f"Listame las carreras que me permitirán desempeñarme como un {role}. Si no tienes carreras aropiadas no inventes respuestas, solo retorna un punto. En caso de que hayan carreras para ser {role}, tienes que incluir el ID y el TITULO\
        en markdown y resume cada uno. Intenta retornar la mayor cantidad de carreras relacionadas."
        response = qa_stuff.run(query)


        review_template = """\
        Para el siguiente texto que está en formato markdown y contiene información de carreras, extrae la siguiente información para cada carrera:
        No inventes carreras nuevas si no tienes en el texto.

        id: ¿Cuál es el id en formato de entero de la carrera? \
        Si esta información no se encuentra, el valor debe ser 0.

        Formatea la salida como lista de ids en formato de entero. De una forma similar a esta '[1, 3, 523, 27]' 

        si no es posible, quiero que retornes una lista vacía '[]'

        texto: {text}
        """

        prompt_template = ChatPromptTemplate.from_template(review_template)

        messages = prompt_template.format_messages(text=response)
        formated_response = self.llm(messages)

        indices = eval(formated_response.content)
        return indices
    

    def add_data(self, current_candidate_response):
        
        if not 'id' in current_candidate_response:
            return 
        id = current_candidate_response['id']
        final_response = current_candidate_response

        dict_full_data_candidate = dict(self.df[self.df['ID'] == id].iloc[0])

        final_response['institucion'] = dict_full_data_candidate['Institucion educativa']
        final_response['origen'] = dict_full_data_candidate['Origen institución educativa']
        final_response['creditos'] = dict_full_data_candidate['Número de créditos programa educativo']
        final_response['metodologia'] = dict_full_data_candidate['Metodología programa educativo']
        final_response['periodicidad'] = dict_full_data_candidate['Periodicidad programa educativo']
        final_response['periodos'] = dict_full_data_candidate['Numero periodos programa educativo']
        final_response['titulo'] = dict_full_data_candidate['Titulo egresado']
        final_response['url'] = dict_full_data_candidate['Link']
        final_response['image_url'] = dict_full_data_candidate['image_url']
        final_response['precio'] = dict_full_data_candidate['precio']

        return final_response

    def get_recommendations(self, role, filtro):
        response = []

        indices_candidatos = []

        try:
            conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
            cur = conn.cursor()
            query = f"SELECT * FROM public.responses_bictia WHERE role = '{role}'"
            
            cur.execute(query)

            results = list(cur.fetchall())

            if len(results) > 0:
                indices_candidatos = [result[1] for result in results]
            else:
                indices_candidatos = self.get_chain_response(role, filtro)

                for indice in indices_candidatos:
                    query = f"INSERT INTO public.responses_bictia (role, education_index) VALUES ('{role}', {indice})"
                    cur.execute(query)

                conn.commit()

        except Error as e:
            print(f"Ocurrió un error: {e}")
            conn.rollback()
            indices_candidatos = self.get_chain_response(role, filtro)
        finally:
            
            cur.close()
            conn.close()

        
        filtered_df = self.df[self.df['ID'].isin(indices_candidatos)]
        carreras_candidatos = list(filtered_df.to_dict('index').values())      

        for candidato in carreras_candidatos:
            respuesta_candidato_actual = {'id':candidato['ID']}
            full_data = self.add_data(respuesta_candidato_actual)
            response.append(full_data) if full_data is not None else None

        return response

def convert_numpy_to_native(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_numpy_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(item) for item in data]
    return data
    
def sanitize_data(data: List[Dict[str, Union[int, float, str]]]) -> List[Dict[str, Union[int, float, str]]]:
    for item in data:
        if isinstance(item.get('creditos'), float) and math.isnan(item['creditos']):
            item['creditos'] = None
    return data

def normalize_query(query):
    reemplazos = {'director técnico':'deportes','director tecnico':'deportes','dermatólogo':'médico', 'pediatra':'médico', 'community manager':'publicidad y mercadeo', 'marketing':'publicidad y mercadeo','cto': 'CHIEF TECHNOLOGY OFFICER gerente de tecnología y programación', 'ceo': 'gerente ejecutivo general', 'cfo': 'gerente de finanzas', 'cgo': 'gerente de crecimiento, relaciones y ventas', 'chef':'gastronomo', 'cocinero':'gastronomo','culinario':'gastronomo','gestor de transito aereo':'controlador aereo', 'project manager':'gestor de proyectos'}
    normalized_query = [ reemplazos[word.lower()] if word.lower() in reemplazos else word for word in query.split()]

    role = ' '.join(normalized_query)
    reemplazos2 = {'director técnico':'director técnico deportes','director tecnico':'director técnico deportes','community manager':'community manager publicidad y mercadeo'}
    if role in reemplazos2:
        role = reemplazos2[role]
    return role

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

recomendador = MajorRecommender()

@app.post("/api/{role}")
def get_role_post(filtro_request: FiltroRequest):

    filtro = dict(filtro_request)
    role = filtro['role']
    role = str(role).lower()
    role = normalize_query(role)
    respuesta = recomendador.get_recommendations(role, filtro)
    native_data = convert_numpy_to_native(respuesta)
    respuesta_compatible = jsonable_encoder(native_data)
    sanitized_data = sanitize_data(respuesta_compatible)
    return {"response":sanitized_data}
