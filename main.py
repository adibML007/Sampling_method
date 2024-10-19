# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

from torchtext.datasets import AG_NEWS
from transformers import AutoModelWithLMHead, AdamW
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

EPOCHS = 50


def preprocess_data(data_iter):
    data = [tokenizer.encode(text) for _, text in data_iter]
    return data


train_iter = AG_NEWS(split='train')
train_data = preprocess_data(train_iter)


model = AutoModelWithLMHead.from_pretrained("gpt2")
optimizer = AdamW(model.parameters())

model.train()
for epoch in range(EPOCHS):
    for batch in train_data:
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


prompt = tokenizer.encode("Write a summary of the new features in the latest release of the Julia Programming Language", return_tensors="pt")
generated = model.generate(prompt)

generated_text = tokenizer.decode(generated[0])
with open("generated.txt", "w") as f:
    f.write(generated_text)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
