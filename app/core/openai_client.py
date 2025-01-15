import openai

from typing import List
from scipy import spatial
from app.config import settings
from app.utils.data_processsing import load_and_process_embeddings
import numpy as np
from scipy.spatial.distance import cosine, euclidean

openai.api_key = settings.OPENAI_API_KEY

df = load_and_process_embeddings()


def handle_chatbot_response(user_message: str) -> str:
    response = answer_question(question=user_message)
    print("response", response)
    return response


def distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine"):
    query_embedding = np.array(query_embedding).flatten()

    distances = []
    for emb in embeddings:
        emb = np.array(emb).flatten()

        if distance_metric == "cosine":
            distances.append(cosine(query_embedding, emb))
        elif distance_metric == "euclidean":
            distances.append(euclidean(query_embedding, emb))
        else:
            raise ValueError("Unknown distance metric: {}".format(distance_metric))

    return distances


def create_context(question, df, max_len=1800, size="ada"):
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]
    print("q_embedding", q_embeddings[:5])
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embedding"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    # Classifique por distância e adicione o texto ao contexto
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])
    return "\n\n###\n\n".join(returns)


def answer_question(
    df=df,
    model="gpt-3.5-turbo",
    question="responda as dúvidas sobre o IFMA?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=None,
    stop_sequence=None,
):
    context = create_context(
        question,
        df=df,
        max_len=max_len,
        size=size,
    )

    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Criar uma conclusão usando a pergunta e o contexto
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Você é um assistente do IFMA. Responda a pergunta abaixo com base no contexto fornecido.\n\nContexto: {context}\n\nPergunta: {question}\n\nResposta:",
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:

        print(f"Erro ao tentar gerar a resposta: {e}")
        return "Infelizmente, não encontrei uma resposta para sua dúvida."
