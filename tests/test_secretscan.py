from opscopilot.secretscan import scan_text


def test_finds_keys():
    fake_google = "AIza" + "a" * 35
    assert scan_text("x.py", f'KEY = "{fake_google}"')
    assert scan_text("x.txt", "token gsk_" + "b" * 24)


def test_placeholders_and_dev_credentials_are_fine():
    ok = "LLM_API_KEY=put-your-key-here\nLLM_API_KEY=ollama\nPOSTGRES_PASSWORD: opscopilot\n"
    assert scan_text(".env.example", ok) == []


def test_real_looking_assignment_is_flagged():
    assert scan_text(".env", "LLM_API_KEY=abcd1234efgh5678")
    assert not scan_text("code.py", "api_key = os.environ[...]")


def test_keys_only_mode_ignores_assignments():
    assert scan_text("log", "PASSWORD=hunter2hunter2", keys_only=True) == []
