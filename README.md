# Candidate Assessment App
powered by watsonx.ai

# Run Locally (windows)
1. Setup virtual environment: `python -m venv .venv`
2. Activate venv: `.venv/Scripts.Activate.ps1`
3. Install Reqs: `pip install -r requirements.txt`
4. Create .env file and populate. Use `/.env.example`
5. Run streamlit app: `streamlit run app.py`

# Manual push ( btw i'm doing it straight sa Code Engine thru Github SSH)
`ibmcloud login --apikey <API_KEY>`
`ibcloud target -g <RESOURCE_GROUP>`
`ibmcloud cr login --client podman` 
`docker tag <local_image> us.icr.io/<my_namespace>/<my_repo>`
`docker push us.icr.io/<my_namespace>/<my_repo>`

