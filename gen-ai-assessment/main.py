import os
import yaml
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import numpy as np



# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client(project="qwiklabs-gcp-01-5813c5344fea")

# TODO: Instantiate a collection reference
collection = db.collection("food-safety")

# TODO: Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004") 

# TODO: Instantiate a Generative AI model here
gen_model = GenerativeModel(model_name="gemini-1.5-pro-001", safety_settings={
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
# TODO: Implement this function to return relevant context
# from your vector database
def search_vector_database(query: str):

    context = ""
    # 1. Generate the embedding of the query
    query_embedding = embedding_model.embed_query(query) # 需要之前定义的 embedding_model
    query_vector = Vector(np.array(query_embedding).astype(np.float32).tolist())

    # 2. Get the 5 nearest neighbors from your collection
    # Call the get() method on the result of your call to
    # find_nearest to retrieve document snapshots.
    results = collection.find_nearest(
        "embedding",  # 直接传递要比较的字段名
        query_vector,
        distance_measure=DistanceMeasure.COSINE,
        limit=5
    ).get()

    # 3. Call to_dict() on each snapshot to load its data.
    # Combine the snapshots into a single string named context
    for doc in results:
        data = doc.to_dict()
        context += data["content"] + "\n\n"

    return context


    # Don't delete this logging statement.
    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )
    return context

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
def ask_gemini(query: str):
  context = search_vector_database(query)
  prompt = f"""
  You are a helpful chatbot that answers questions about food safety.
  Answer the following question based on the context provided.
  If you cannot answer the question based on the context, say "I am not able to answer this question".

  Question: {query}

  Context:
  {context}
  """

  response = gen_model.generate_content(prompt)
  return response.text

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
