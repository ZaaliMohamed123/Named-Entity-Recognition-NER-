# Named Entity Recognition (NER) Models

## Description
This project implements various Named Entity Recognition (NER) models to identify and classify named entities in text. The application provides a user-friendly interface using Streamlit, allowing users to input text and receive predictions from different NER models.

## Installation
To set up the project, ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To run the Streamlit application, execute the following command:

```bash
streamlit run app/streamlitapp.py
```

Once the application is running, you can enter text in the provided input area and click the "Predict" button to see the predictions from various models.

## Models Used
- **Model 1**: A Keras-based NER model that predicts entities such as Geographical Entities, Organizations, and Persons.
- **Model 2**: Another Keras-based NER model that identifies entities like Miscellaneous, Persons, and Locations.
- **SpaCy Model**: Utilizes the SpaCy library to predict entities in the input text.
- **Stanza Model**: Uses the Stanza library for entity recognition.

## Contributors
Made by [ZAALI Mohamed](https://www.linkedin.com/in/m-zaali/) and [SEKAL Douaâ](https://www.linkedin.com/in/douaa-sekal/).
