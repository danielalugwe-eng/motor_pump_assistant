from importlib import import_module


def test_api_module_imports_without_openai_key():
    module = import_module("src.main")

    assert module.app.title == "Motor Pump Predictive System"
    assert callable(module.root)
