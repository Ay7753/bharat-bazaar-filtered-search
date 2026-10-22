# Failures

Record failed experiments, edge cases, and fixes here.

## Environment check

`pip check` reports pre-existing conflicts in the global environment: Django 4.1.13 is below django-extensions' requirement; FastAPI, python-multipart, and Starlette are below Gradio's requirements; Click is too new for gTTS; and google-ai-generativelanguage is below langchain-google-genai's requirement. These are unrelated to this project's NumPy and scikit-learn install.