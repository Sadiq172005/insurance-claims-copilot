install:  ; pip install -e ".[all]"
samples:  ; python generate_samples.py
test:     ; python test_validators.py
index:    ; python -c "from pipeline import build_index; build_index()"
eval:     ; python evaluate.py
eval-offline: ; OFFLINE_MODE=1 python evaluate.py
notebook: ; python build_notebook.py --execute
api:      ; uvicorn app:app --reload
ui:       ; streamlit run streamlit_app.py
docker:   ; docker build -t claims-copilot . && docker run -p 8000:8000 --env-file .env claims-copilot
deploy:   ; bash deploy/aws_deploy.sh
