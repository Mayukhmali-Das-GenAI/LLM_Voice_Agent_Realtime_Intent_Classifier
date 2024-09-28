# LLM_Voice_Agent_Realtime_Intent_Classifier

## Why do we need this?

In today's fast-paced, AI-integrated world, conversations often involve a dynamic mix of queries directed at AI language models (LLMs) and statements meant for human interaction. A real-time LLM Intent Classifier is crucial for instantaneously distinguishing between these, ensuring that AI assistants respond appropriately and don't interject in human-to-human communication as it happens.

Consider these real-time conversation snippets:

1. Bakery scenario:
   ```
   ["Locate a nearby bakery.", "LLM"],
   ["Do you have gluten-free options?", "Non-LLM"],
   ["Add 'bread' to my shopping list.", "LLM"],
   ["Check recipes for gluten-free bread.", "LLM"],
   ["I'll take a loaf, please.", "Non-LLM"]
   ```

2. Gym scenario:
   ```
   ["Locate the closest gym.", "LLM"],
   ["Hello, can I get a day pass?", "Non-LLM"],
   ["Play my workout playlist.", "LLM"],
   ["Track my calories burned.", "LLM"],
   ["Great, here's my ID.", "Non-LLM"]
   ```

3. Mixed communication scenario:
   ```
   ["Mom, I'll be home late tonight.", "Non-LLM"],
   ["Find the nearest coffee shop.", "LLM"],
   ["What's the traffic like on the way to downtown?", "LLM"]
   ```

In these examples, the real-time classifier must instantly decide whether each statement should be processed by the LLM or ignored, allowing for seamless integration of AI assistance into live conversations without disrupting normal human interactions.

A Real-time LLM Intent Classifier enables:
- Instant response to AI-directed queries
- Immediate filtering of human-to-human communication
- Seamless switching between AI and human interaction modes
- Enhanced user experience in mixed human-AI environments
- Prevention of awkward or unnecessary AI responses in real-time dialogue


## Overview

This project implements an end-to-end solution for training and deploying a very low latency model that classifies user intents in real-time as either "LLM" (to be answered by an AI language model) or "Non-LLM" (not to be answered by an AI).

The system consists of two main components:

1. A BERT-based intent classification model optimized for real-time inference
2. A Streamlit-based demo application showcasing real-time classification. Realtime audio transcription is done using OpenAI Whisper.

The intent classifier is trained on a dataset of conversation segments, each labeled as either "LLM" or "Non-LLM". The model learns to distinguish between queries that should be answered by an AI language model and those that should not, all in real-time.

## Dataset

### Current Dataset
The current dataset was created using GPT-o1 preview. It consists of a diverse range of conversation snippets that simulate real-world interactions in various scenarios. This synthetic dataset provides a solid foundation for training my real-time intent classifier. I plan on training on a larger dataset in the future.

### Future Dataset Expansion Plans
I have exciting plans to expand and enhance this dataset:

1. **Real-time Audio Dataset**: I aim to incorporate a real-time audio dataset to capture additional features such as:
   - Sound localization
   - Audio cues
   - Prosody and intonation
   - Background noise characteristics

2. **Multimodal Data**: Future versions may include visual cues (if applicable) to provide a more comprehensive context for intent classification.

3. **Diverse Scenarios**: I plan to continuously add more diverse and complex conversational scenarios to improve the model's robustness.

These expansions will allow my real-time intent model to leverage a richer set of features, potentially improving its accuracy and applicability in various real-world settings.


## Installation

Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Prepare your training data in JSON format (see `training_data/intent_llm.json` for an example).
2. Run the training script:
   
   ```
   python train_realtime_intent_model.py
   ```

The script will:
- Load and preprocess the data
- Split it into training and test sets
- Train a BERT model optimized for real-time classification
- Evaluate the model's performance
- Save the trained model

#### Training Results

I trained the base model for 3 epochs.

```
Epoch 3/3
Epoch 3 completed. Average Training Loss: 0.0313
Validation Loss: 0.0649
Precision: 0.9681, Recall: 1.0000, F1 Score: 0.9838, Event Rate: 0.3957

Example Prediction:
[LLM] What's the exchange rate for euros to dollars?
[Non-LLM] Coming mom give me 5 mins
[LLM] Make a basic itinerary for my Europe trip

```


### Running the Real-time Demo

1. Ensure you have the trained model in the `intent_model` directory.
2. Start the Streamlit app:
   ```
   streamlit run realtime_demo_app.py
   ```

3. Use the web interface to:
   - Record live audio input
   - See real-time transcription of speech to text
   - View instant intent classification results
   - Monitor confidence scores in real-time
  
![image](https://github.com/user-attachments/assets/a8a08fc2-dde1-4411-83be-79cd550859d9)

