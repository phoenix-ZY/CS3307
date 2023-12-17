from transformers import DistilBertForSequenceClassification, AdamW
import torch
from utils import *
from tqdm import tqdm
def bert_train(num_epochs,model,train_dataloader,valid_dataloader,device,optimizer,batch_size,load_path= "",save_path= ""):
    if load_path:
        model = DistilBertForSequenceClassification.from_pretrained(load_path, num_labels=2)
    model = model.to(device)
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
            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))


        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss}')
        right = 0
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
                print(predictions)
                print(labels)
                right += now_right
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))

        valid_accuracy = float(right)/(float(len(valid_dataloader) * batch_size))
        print(f'Epoch {epoch+1}/{num_epochs} - Valid Accuracy: {valid_accuracy}')
    if(save_path):
        model.save_pretrained(save_path)

def train(num_epochs,model,train_dataloader,valid_dataloader,device,optimizer,criterion,batch_size,load_path = "",save_path = ""):
    best_valid_loss = float('inf')
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    for epoch in range(num_epochs):
        loop = tqdm(train_dataloader, total =len(train_dataloader))
        total_loss = 0
        # 训练
        model.train()
        times = 0
        right = 0
        for batch in loop:
            optimizer.zero_grad()
            times +=1
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)

            model.zero_grad()
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            now_right =  (torch.round(torch.sigmoid(outputs)) == labels).sum()
            right += now_right
            if times %50 == 0:
                loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss}')

        # 验证
        model.eval()
        y_true = []
        y_pred = []
        times = 0
        right = 0
        total_loss = 0
        loop = tqdm(valid_dataloader, total =len(valid_dataloader))
        with torch.no_grad():
            for batch in loop:
                input_ids = batch['sentence'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).float().unsqueeze(1)

                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                now_right =  (torch.round(torch.sigmoid(outputs)) == labels).sum()
                right += now_right
                loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
                loop.set_postfix(loss=loss, acc=float(now_right)/float(batch_size))

        valid_accuracy = float(right)/(float(len(valid_dataloader) * batch_size))
        print(f'Epoch {epoch+1}/{num_epochs} - Valid Accuracy: {valid_accuracy}')
        avg_loss = total_loss / len(train_dataloader)
        if avg_loss < best_valid_loss and save_path:
            best_valid_loss = avg_loss
            torch.save(model.state_dict(), save_path)





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

def test(test_dataloader,device,model, load_path):
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)

            outputs = model(input_ids)
            outputs =  torch.round(torch.sigmoid(outputs))
            y_true.extend(labels.tolist())
            y_pred.extend(outputs.tolist())

    valid_accuracy = accuracy_score(y_true, y_pred)
    print(f'test Accuracy: {valid_accuracy}')

def save_results(test_dataloader,device,model_name):
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    intermediate_outputs = {}
    def save_intermediate_results(module, input, output):
        intermediate_outputs[module] = output
    model.eval()
    batch_size = 16
    device = "cpu"
    layer_hooks = []
    for number in range(len(model.distilbert.transformer.layer)):
        module = model.distilbert.transformer.layer[number]
        hook  = module.register_forward_hook(save_intermediate_results)
        layer_hooks.append(hook)
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            break
    number = 0
    for name,output in intermediate_outputs.items():
        print(output[0].shape)
        torch.save(output, f"results/middleresults/transfomer_{number}output.pt")
        number += 1

    for hook in layer_hooks:
        hook.remove()

