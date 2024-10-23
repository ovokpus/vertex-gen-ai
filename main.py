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

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# TODO: Instantiate a collection reference
collection = db.collection('food-safety')

# TODO: Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

# TODO: Instantiate a Generative AI model here
gen_model = GenerativeModel("gemini-pro")

# TODO: Implement this function to return relevant context
# from your vector database
def search_vector_database(query: str):

    context = ""

    # 1. Generate the embedding of the query
    query_embedding = embedding_model.embed_query(query)

    # 2. Get the 5 nearest neighbors from your collection.
    # Call the get() method on the result of your call to
    # find_neighbors to retrieve document snapshots.

    # Perform the vector query using 'find_nearest'
    results = collection.find_nearest(
        vector_field='embedding',
        query_vector=Vector(query_embedding),
        limit=5,
        distance_measure=DistanceMeasure.COSINE
    ).get()

    # 3. Call to_dict() on each snapshot to load its data.
    # Combine the snapshots into a single string named context
    for doc in results:
        doc_dict = doc.to_dict()
        context += doc_dict.get('content', '') + "\n"

    # Don't delete this logging statement.
    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )
    return context

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
def ask_gemini(question):

    # 1. Create a prompt_template with instructions to the model
    # to use provided context info to answer the question.
    prompt_template = (
        "Answer the following question based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    # 2. Use your search_vector_database function to retrieve context
    # relevant to the question.
    context = search_vector_database(question)
    
    # 3. Format the prompt template with the question & context
    prompt = prompt_template.format(context=context, question=question)

    # 4. Set Gemini's Dangerous Content safety setting to block only high-probability dangerous content
    safety_settings = [
        {
            "category": "harm",
            "threshold": "HIGH"
        }
    ]

    # 5. Pass the complete prompt template to gemini and get the text
    # of its response to return below.
    # Since we need to set the temperature to 0, we'll pass it in the predict method.
    response = gen_model.generate_content(
                    prompt, 
                    generation_config={"temperature": 0}
                    )

    return response

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
        response = ask_gemini(question)
        answer = response.candidates[0].content.parts[0].text

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + str(answer))

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
