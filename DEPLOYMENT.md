# Streamlit Community Cloud Deployment

Deploy settings:

```text
Main file path: app.py
Python dependencies: requirements.txt
System packages: packages.txt
```

Required model artifacts:

```text
tokenizer.pkl
lstm_model.h5
cnn_model.h5
```

Place those files either in the project root or in the `models/` folder before deploying.

## Steps

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from that repository.
3. Select the branch that contains this project.
4. Set the main file path to `app.py`.
5. Deploy.

The app will open without the model files, but predictions will show a missing-file message until the artifacts are added.
