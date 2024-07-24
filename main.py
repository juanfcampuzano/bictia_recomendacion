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
from fastapi.encoders import jsonable_encoder
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Union
import math

load_dotenv()

class MajorRecommender:

    def __init__(self):
        self.qa_stuff = None
        self.llm = None
        self.prepare_qa_chain()
        self.df = pd.read_csv('ProgramasCompletos.csv')
        
    def prepare_qa_chain(self):

        loader = CSVLoader("CarrerasIndice.csv", encoding='utf-8')
        docs = loader.load()

        embeddings = OpenAIEmbeddings(openai_api_key= os.getenv('OPENAI_API_KEY'))
        db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
        )
        self.llm = ChatOpenAI(temperature = 0.0, openai_api_key= os.getenv('OPENAI_API_KEY'))
        retriever = db.as_retriever(search_kwargs={"k": 15})

        self.qa_stuff = RetrievalQA.from_chain_type(
        llm=self.llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
        )

    def get_chain_response(self, role):
        query =  f"Listame las carreras que me permitirán desempeñarme como un {role}. Si no tienes carreras aropiadas no inventes respuestas, solo retorna un punto. En caso de que hayan carreras para ser {role}, tienes que incluir el ID y el TITULO\
        en markdown y resume cada uno. Intenta retornar la mayor cantidad de carreras relacionadas."
        response = self.qa_stuff.run(query)


        review_template = """\
        Para el siguiente texto que está en formato markdown y contiene información de carreras, extrae la siguiente información para cada carrera:
        No inventes carreras nuevas si no tienes en el texto.

        id: ¿Cuál es el id de la carrera? \
        Si esta información no se encuentra, el valor debe ser -1.

        Formatea la salida como lista de ids. De una forma similar a esta '[1, 123, 523]'

        texto: {text}
        """

        prompt_template = ChatPromptTemplate.from_template(review_template)

        messages = prompt_template.format_messages(text=response)
        formated_response = self.llm(messages)

        response = eval(formated_response.content)

        filtered_df = self.df[self.df['ID'].isin(response)]

        carreras_afines = list(filtered_df.to_dict('index').values())
        return carreras_afines
    
    def get_recommendation_percentage(self, role, programa):
        template_recommendation_percentage =  """Quiero llegar a ser un {role}, necesito que me ofrezcas la afinidad (porcentaje) para el programa que te doy, basate en calidad de institucion (ranking el cual considera entre menor valor mejor, desde 1 hasta 200), Opiniones institución educativa. Tienes que incluir el ID, PORCENTAJE de afinidad y el MOTIVO de la recomendación\
            en markdown. PROGRAMA = {programa}

            id: ¿Cuál es el id de la carrera? \

            afinidad: ¿De 0 a 100 que afinidad me das a estudiar esta carrera en esta universidad? \
            
            motivo: ¿Por qué me das este porcentaje de afinidad para estudiar ese programa? No menciones directamente el ranking, en caso de que sea una institucion buena, resaltalo. Considera muy buena entre el raking 1 hasta el 15. Si la institucion no tiene un raking bueno, no lo menciones.\

            Formatea la salida como un diccionario, no como un json, las keys deben ser id, afinidad y motivo.
            """

        prompt_template = ChatPromptTemplate.from_template(template_recommendation_percentage)  

        messages = prompt_template.format_messages(role = role, programa = programa)
        formated_response = self.llm(messages)

        response = eval(formated_response.content.replace("```", "").replace("json", ""))

        final_response = {k.lower():v for k,v in response.items()}
        return final_response
    
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

        return final_response

    def get_recommendations(self, role):
        response = []

        carreras_candidatos = self.get_chain_response(role)


        for candidato in carreras_candidatos:
            respuesta_candidato_actual = self.get_recommendation_percentage(role, candidato)
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

app = FastAPI()
recomendador = MajorRecommender()

@app.get("/api/test")
def test():
    return {"response":"test"}

@app.get("/api/{role}")
def get_role(role):
    respuesta = recomendador.get_recommendations(role)
    native_data = convert_numpy_to_native(respuesta)
    respuesta_compatible = jsonable_encoder(native_data)
    respuesta_compatible.sort(key=lambda x: -x["afinidad"])
    print('respuesta_compatible', respuesta_compatible)
    sanitized_data = sanitize_data(respuesta_compatible)
    print('sanitized_data', sanitized_data)
    return {"response":sanitized_data}