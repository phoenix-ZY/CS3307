from transformers import DistilBertForSequenceClassification, AdamW
import torch
from utils import *
from tqdm import tqdm
def bert_train(num_epochs,model,train_dataloader,valid_dataloader,device,optimizer,batch_size,save_path):
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader, total =len(train_dataloader))
        total_loss = 0

        # 训练
        model.train()
        times = 0
        right = 0
        for batch in loop:
            times +=1
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            model.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            now_right =  (predictions == labels).sum()
            right += now_right
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))


        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss}')

        # 验证
        model.eval()
        y_true = []
        y_pred = []
        times = 0
        loop = tqdm(valid_dataloader, total =len(valid_dataloader))
        with torch.no_grad():
            for batch in loop:
                input_ids = batch['sentence'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                now_right =  (predictions == labels).sum()
                right += now_right
                y_true.extend(labels.tolist())
                y_pred.extend(predictions.tolist())

                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))

        valid_accuracy = float(right)/float(len(valid_dataloader))
        print(f'Epoch {epoch+1}/{num_epochs} - Valid Accuracy: {valid_accuracy}')
    if(save_path):
        model.save_pretrained(save_path)

def train(num_epochs,model,train_dataloader,valid_dataloader,device,optimizer,batch_size,save_path):
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader, total =len(train_dataloader))
        total_loss = 0

        # 训练
        model.train()
        times = 0
        right = 0
        for batch in loop:
            times +=1
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            model.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            now_right =  (predictions == labels).sum()
            right += now_right
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))


        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss}')

        # 验证
        model.eval()
        y_true = []
        y_pred = []
        times = 0
        loop = tqdm(valid_dataloader, total =len(valid_dataloader))
        with torch.no_grad():
            for batch in loop:
                input_ids = batch['sentence'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                now_right =  (predictions == labels).sum()
                right += now_right
                y_true.extend(labels.tolist())
                y_pred.extend(predictions.tolist())

                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))

        valid_accuracy = float(right)/float(len(valid_dataloader))
        print(f'Epoch {epoch+1}/{num_epochs} - Valid Accuracy: {valid_accuracy}')
    if(save_path):
        model.save_pretrained(save_path)





def bert_test(test_dataloader,device,model_name):
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    valid_accuracy = accuracy_score(y_true, y_pred)
    print(f'test Accuracy: {valid_accuracy}')