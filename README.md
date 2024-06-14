# Disclaimer
The styling is horrible currently. We can adjust everything by using HTML + CSS.
# Preperation
I use Python 3.10.11. Other up-to-date version should work as well

Create config.py and add a variable in this format:
```python
OPENAI_API_KEY = "YOUR-KEY"
ASSISTANT_ID = "YOUR-ASSISTANT-ID" # or None
```

# Model Interface
the CNN (or whatever) will later be accessed via model_interface.py 

We can use them as classical functions or as wrapper functions if we want to use class methods

Futher information can be found here: [OPENAI FUNCTION CALLING DOCS](https://platform.openai.com/docs/guides/function-calling)
# Run
entrance file is: main.py
```shell
streamlit run main.py
```

# LINKS
[TIPI](https://gosling.psy.utexas.edu/wp-content/uploads/2014/09/JRP-03-tipi.pdf)