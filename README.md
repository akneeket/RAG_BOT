

---

# RAG Bot

RAG Bot is a news research tool that allows users to retrieve relevant information from news articles based on their questions. The tool works by processing URLs of articles or PDFs and then answering user queries with insights from the articles.

## Features

- Upload PDFs or enter URLs of news articles.
- Process the content of the articles using LangChain’s UnstructuredURL Loader.
- Use OpenAI's embeddings for text similarity.
- Retrieve relevant answers quickly using FAISS.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/akneeket/RAG_BOT.git
   ```

2. Navigate to the project directory:

   ```
   cd RAG_BOT
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the app using Streamlit:

   ```
   streamlit run app.py
   ```

2. Enter a URL or upload a PDF file.
3. Ask questions related to the article or PDF, and get answers based on its content.

## Project Structure

- `app.py`: The main Streamlit application script.
- `requirements.txt`: Lists the required Python packages for the project.
- `faiss_index.pkl`: Stores the FAISS index for faster search.
- `README.md`: This file.

## License

This project is licensed under the MIT License.

---

