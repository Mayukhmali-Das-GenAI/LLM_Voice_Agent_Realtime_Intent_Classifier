import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import pandas as pd

class IntentDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]['text']
        label = self.examples[idx]['label']
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        loop = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', pos_label=1)
    event_rate = sum(true_labels) / len(true_labels)
    return avg_loss, precision, recall, f1, event_rate

def main():
    print("Loading data from 'intent_llm.json'...")
    with open('training_data/intent_llm.json', 'r') as f:
        data = json.load(f)
    
    print("Preparing training examples...")
    examples = []
    for conversation in data:
        segments = conversation['Segments']
        context_sentences = []
        for segment in segments:
            text, segment_type = segment
            label = 0 if segment_type == 'LLM' else 1
            context = ' '.join(context_sentences)
            input_text = context + ' ' + text if context else text
            examples.append({'text': input_text.strip(), 'label': label})
            context_sentences.append(text)
    
    print(f"Total examples prepared: {len(examples)}")
    
    print("Splitting data into training and test sets...")
    train_examples, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
    print(f"Training examples: {len(train_examples)}, Test examples: {len(test_examples)}")
    
    print("Saving training data to 'training_data.xlsx'...")
    train_df = pd.DataFrame(train_examples)
    train_df.to_excel('training_data.xlsx', index=False)
    
    print("Saving test data to 'test_data.xlsx'...")
    test_df = pd.DataFrame(test_examples)
    test_df.to_excel('test_data.xlsx', index=False)
    
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Creating dataset objects...")
    train_dataset = IntentDataset(train_examples, tokenizer)
    test_dataset = IntentDataset(test_examples, tokenizer)
    
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    print("Initializing BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    epochs = 3
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        loop = tqdm(train_loader, desc="Training", leave=False)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_loss:.4f}")
        
        val_loss, precision, recall, f1, event_rate = evaluate(model, test_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Event Rate: {event_rate:.4f}")
    
    print("\nSaving the trained model...")
    model.save_pretrained('intent_model')
    tokenizer.save_pretrained('intent_model')
    print("Model saved to 'intent_model/' directory.")
    
    print("\nStarting final evaluation on the test set...")
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating", leave=False)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', pos_label=1)
    event_rate = sum(true_labels) / len(true_labels)
    print(f"Final Evaluation completed. Accuracy on test set: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Event Rate: {event_rate:.4f}")
    
    print("\nExample Prediction:")
    sentences = [
        "What's the exchange rate for euros to dollars?",
        "Coming mom give me 5 mins",
        "Make a basic itinerary for my Europe trip"
    ]
    context_sentences = []
    for idx, sentence in enumerate(sentences, 1):
        context = ' '.join(context_sentences)
        input_text = context + ' ' + sentence if context else sentence
        encoding = tokenizer(
            input_text.strip(),
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        label = 'LLM' if prediction == 0 else 'Non-LLM'
        print(f"[{label}] {sentence}")
        context_sentences.append(sentence)

if __name__ == '__main__':
    main()