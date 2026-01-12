PYTHON := python3
VENV_ACTIVATE := source .venv/bin/activate
FASTAPI_APP := api_inference:app
STREAMLIT_APP := streamlit_frontend.py
PORT := 8000

.PHONY: all backend frontend clean help

all: backend frontend
	@echo "Beide Services (Backend und Frontend) wurden gestartet (möglicherweise in separaten Terminals)."

backend:
	@echo "Starte FastAPI-Backend mit Uvicorn..."
	# Aktiviert die virtuelle Umgebung und startet Uvicorn
	$(VENV_ACTIVATE) && uvicorn $(FASTAPI_APP) --reload --port $(PORT)


frontend:
	@echo "Starte Streamlit-Frontend..."
	# Aktiviert die virtuelle Umgebung und startet Streamlit
	$(VENV_ACTIVATE) && streamlit run $(STREAMLIT_APP)


clean:
	@echo "Entferne Python-Cache-Dateien..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "HINWEIS: Sie müssen den Graphen-Cache (cache/product_graph.pt) ggf. manuell löschen."

help:
	@echo "Verfügbare Befehle:"
	@echo "  make backend   - Startet den FastAPI-Server (Backend) mit --reload."
	@echo "  make frontend  - Startet die Streamlit-Anwendung (Frontend)."
	@echo "  make all       - Führt 'make backend' und 'make frontend' aus (Achtung: benötigt zwei Terminals)."
	@echo "  make clean     - Löscht Python-Cache-Dateien."