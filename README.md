# news-summarizer
venv/Scripts/Activate.ps1

pip install -r requirements.txt

run back end:
uvicorn backend.app:app --reload --port 8000

run streamlit
streamlit run frontend/streamlit_app.py
