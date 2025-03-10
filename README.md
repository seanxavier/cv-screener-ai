# Candidate Assessment App
powered by watsonx.ai

# Run Locally (windows)
1. Setup virtual environment: `python -m venv .venv`
2. Activate venv: `.venv/Scripts.Activate.ps1`
3. Install Reqs: `pip install -r requirements.txt`
4. Create .env file and populate. Use `/.env.example`
5. Run streamlit app: `streamlit run app.py`

# Run Locally (Mac)
1. Setup virtual environment: `python3.12 -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Install Reqs: `pip3.12 install -r requirements.txt`
4. Create .env file and populate. Use `/.env.example`
5. Run streamlit app: `streamlit run app.py`

# Manual push ( btw i'm doing it straight sa Code Engine thru Github SSH)
`ibmcloud login --apikey <API_KEY>`
`ibcloud target -g <RESOURCE_GROUP>`
`ibmcloud cr login --client podman` 
`docker tag <local_image> us.icr.io/<my_namespace>/<my_repo>`
`docker push us.icr.io/<my_namespace>/<my_repo>`

`ibmcloud login --apikey <API_KEY>`
`ibcloud target -g <RESOURCE_GROUP>`
`ibmcloud cr login --client podman` 
`podman build -t <image_name> .`
`podman tag <local_image> us.icr.io/<my_namespace>/<my_repo>`
`podman push us.icr.io/<my_namespace>/<my_repo>`


