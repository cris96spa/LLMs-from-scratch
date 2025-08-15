import tiktoken

if __name__ == "__main__":
    with open("data/the-verdict.txt") as f:
        text = f.read()
    encoding = tiktoken.encoding_for_model("")
