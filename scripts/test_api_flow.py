from src.main import ask


class Q:
    def __init__(self, text: str):
        self.text = text


if __name__ == "__main__":
    print(ask(Q("How do I maintain the pump?")))
